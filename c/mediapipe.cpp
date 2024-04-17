#include "mediapipe.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/tool/options_util.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // MEDIAPIPE_DISABLE_GPU


#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"

#ifdef __ANDROID__
#   include "mediapipe/util/android/asset_manager_util.h"
#   include "mediapipe/framework/port/singleton.h"
#endif

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "google/protobuf/util/json_util.h"

#include <string>
#include <cstring>
#include <variant>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cstdlib>

#ifndef __ANDROID__
ABSL_DECLARE_FLAG(std::string, resource_root_dir);
#endif

thread_local absl::Status last_error;

struct mp_node_option {
    const char* node;
    const char* option;
    std::variant<float, double> value;
};

struct mp_instance_builder {
    const char* graph_filename;
    const char* input_stream;
    std::vector<mp_node_option> options;
    std::map<std::string, mediapipe::Packet> side_packets;
};

struct mp_instance {
    mediapipe::CalculatorGraph graph;
#if !MEDIAPIPE_DISABLE_GPU
    mediapipe::GlCalculatorHelper gpu_helper;
#endif  // !MEDIAPIPE_DISABLE_GPU
    std::string input_stream;
    size_t frame_timestamp;
};

struct mp_poller {
    mediapipe::OutputStreamPoller poller;
};

struct mp_packet {
    mediapipe::Packet packet;
};

template<typename List, typename Landmark>
static mp_pose_landmark_list* get_pose_landmarks(mp_packet* packet) {
    const auto& mp_list = packet->packet.template Get<List>();

    auto* lists = new mp_landmark_list[1];

    auto* list = new mp_landmark[mp_list.landmark_size()];

    for (int j = 0; j < mp_list.landmark_size(); j++) {
        const Landmark& mp_landmark = mp_list.landmark(j);
        list[j] = {
            mp_landmark.x(),
            mp_landmark.y(),
            mp_landmark.z()
        };
    }

    lists[0] = mp_landmark_list {
        list,
        (int) mp_list.landmark_size()
    };

    return new mp_pose_landmark_list {
        lists,
        (int) 1
    };
}

template<typename Rect>
static mp_rect_list* get_rects(mp_packet* packet) {
    const auto& mp_data = packet->packet.template Get<std::vector<Rect>>();
    auto* list = new mp_rect[mp_data.size()];

    for (int i = 0; i < mp_data.size(); i++) {
        const Rect& mp_rect = mp_data[i];
        list[i] = {
            (float) mp_rect.x_center(),
            (float) mp_rect.y_center(),
            (float) mp_rect.width(),
            (float) mp_rect.height(),
            mp_rect.rotation(),
            mp_rect.rect_id()
        };
    }

    return new mp_rect_list {
        list,
        (int) mp_data.size()
    };
}

extern "C" {

MEDIAPIPE_API mp_instance_builder* mp_create_instance_builder(const char* graph_filename, const char* input_stream) {
    return new mp_instance_builder { graph_filename, input_stream, {} };
}

MEDIAPIPE_API void mp_add_option_float(mp_instance_builder* instance_builder, const char* node, const char* option, float value) {
    instance_builder->options.push_back({ node, option, value });
}

MEDIAPIPE_API void mp_add_option_double(mp_instance_builder* instance_builder, const char* node, const char* option, double value) {
    instance_builder->options.push_back({ node, option, value });
}

MEDIAPIPE_API void mp_add_side_packet(mp_instance_builder* instance_builder, const char* name, mp_packet* packet) {
    instance_builder->side_packets.insert({name, packet->packet});
    mp_destroy_packet(packet);
}

MEDIAPIPE_API mp_instance* mp_create_instance(mp_instance_builder* builder) {
    mediapipe::CalculatorGraphConfig config;

    std::ifstream stream(builder->graph_filename, std::ios::binary | std::ios::ate);
    if (!stream) {
        last_error = absl::Status(absl::StatusCode::kNotFound, "Failed to open graph file");
        return nullptr;
    }
    
    size_t size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    char* memory = new char[size];
    stream.read(memory, size);
    config.ParseFromArray(memory, size);
    delete[] memory;

    mediapipe::ValidatedGraphConfig validated_config;
    validated_config.Initialize(config);
    mediapipe::CalculatorGraphConfig canonical_config = validated_config.Config();

    for (const mp_node_option& option : builder->options) {
        for (auto& node : *canonical_config.mutable_node()) {
            if (node.name() != option.node) {
                continue;
            }

            google::protobuf::Message* ext;

            if (node.calculator() == "ThresholdingCalculator")
                ext = node.mutable_options()->MutableExtension(mediapipe::ThresholdingCalculatorOptions::ext);
            else if (node.calculator() == "TensorsToDetectionsCalculator")
                ext = node.mutable_options()->MutableExtension(mediapipe::TensorsToDetectionsCalculatorOptions::ext);
            else {
                assert(!"Unknown node calculator");
                return nullptr;
            }

            auto* descriptor = ext->GetDescriptor();
            auto* reflection = ext->GetReflection();
            auto* field_descriptor = descriptor->FindFieldByName(option.option);

            switch (option.value.index()) {
                case 0: reflection->SetFloat(ext, field_descriptor, std::get<0>(option.value)); break;
                case 1: reflection->SetDouble(ext, field_descriptor, std::get<1>(option.value)); break;
            }
        }
    }

    /* For printing the graph

    google::protobuf::util::JsonPrintOptions json_options;
    json_options.add_whitespace = true;

    std::string str;
    google::protobuf::util::MessageToJsonString(canonical_config, &str, json_options);
    std::cout << str << std::endl;
    
    */
    auto* instance = new mp_instance;
    absl::Status result = instance->graph.Initialize(canonical_config, builder->side_packets);
    
    if (!result.ok()) {
        last_error = result;
        return nullptr;
    }

    instance->input_stream = builder->input_stream;
    instance->frame_timestamp = 0;

    delete builder;
    return instance;
}

#if !MEDIAPIPE_DISABLE_GPU
MEDIAPIPE_API mp_instance* mp_initialize_gpu(mp_instance_builder* builder, mp_instance *instance) {
    absl::StatusOr<std::shared_ptr<mediapipe::GpuResources>> gpu_resources = mediapipe::GpuResources::Create();

    absl::Status result = instance->graph.SetGpuResources(std::move(*gpu_resources));
    if (!result.ok()) {
        last_error = result;
        return nullptr;
    }

    instance->gpu_helper.InitializeForTest(instance->graph.GetGpuResources().get());
    return instance;
}
#endif  // MEDIAPIPE_DISABLE_GPU

MEDIAPIPE_API mp_poller* mp_create_poller(mp_instance* instance, const char* output_stream) {
    absl::StatusOr<mediapipe::OutputStreamPoller> result = instance->graph.AddOutputStreamPoller(output_stream);
    if (!result.ok()) {
        last_error = result.status();
        return nullptr;
    }

    return new mp_poller {
        std::move(*result)
    };
}

MEDIAPIPE_API bool mp_start(mp_instance* instance) {
    absl::Status result = instance->graph.StartRun({});

    if (!result.ok()) {
        last_error = result;
        return false;
    }

    return true;
}

MEDIAPIPE_API bool mp_process(mp_instance* instance, mp_packet* packet) {
    mediapipe::Timestamp mp_timestamp(instance->frame_timestamp++);
    mediapipe::Packet mp_packet = packet->packet.At(mp_timestamp);
    auto result = instance->graph.AddPacketToInputStream(instance->input_stream, mp_packet);
    mp_destroy_packet(packet);

    if (!result.ok()) {
        last_error = result;
        return false;
    }

    return true;
}

MEDIAPIPE_API bool mp_process_gpu(mp_instance* instance, mp_packet* packet) {
    absl::Status result = instance->gpu_helper.RunInGlContext([&instance, &packet]() -> absl::Status {
        mediapipe::Timestamp mp_timestamp(instance->frame_timestamp++);
        mediapipe::Packet mp_packet = packet->packet.At(mp_timestamp);
        auto result = instance->graph.AddPacketToInputStream(instance->input_stream, mp_packet);
        mp_destroy_packet(packet);

        if (!result.ok()) {
            last_error = result;
            return last_error;
        }

        return absl::OkStatus();
    });
    if (!result.ok()) {
        return false;
    }
    return true;
}

MEDIAPIPE_API bool mp_wait_until_idle(mp_instance* instance) {
    absl::Status result = instance->graph.WaitUntilIdle();

    if (!result.ok()) {
        last_error = result;
        return false;
    }

    return true;
}

MEDIAPIPE_API bool mp_wait_until_idle_gpu(mp_instance* instance) {
    absl::Status result = instance->gpu_helper.RunInGlContext([&instance]() -> absl::Status {
        auto result = instance->graph.WaitUntilIdle();
        if (!result.ok()) {
            last_error = result;
            return last_error;
        }

        return absl::OkStatus();
    });
    if (!result.ok()) {
        return false;
    }
    return true;
}

MEDIAPIPE_API int mp_get_queue_size(mp_poller* poller) {
    return poller->poller.QueueSize();
}

MEDIAPIPE_API void mp_destroy_poller(mp_poller* poller) {
    delete poller;
}

MEDIAPIPE_API bool mp_destroy_instance(mp_instance* instance) {
    absl::Status result = instance->graph.CloseInputStream(instance->input_stream);
    if (!result.ok()) {
        last_error = result;
        return false;
    }

    result = instance->graph.WaitUntilDone();
    if (!result.ok()) {
        last_error = result;
        return false;
    }

    delete instance;
    return true;
}

MEDIAPIPE_API void mp_set_resource_dir(const char* dir) {
#ifndef __ANDROID__
    absl::SetFlag(&FLAGS_resource_root_dir, dir);
#endif
}

#ifdef __ANDROID__
MEDIAPIPE_API void mp_init_asset_manager(JNIEnv* env, jobject android_context, jstring cache_dir_path) {
    mediapipe::AssetManager* asset_manager = Singleton<mediapipe::AssetManager>::get();
    const char* c_cache_dir_path = env->GetStringUTFChars(cache_dir_path, nullptr);
    asset_manager->InitializeFromActivity(env, android_context, c_cache_dir_path);
    env->ReleaseStringUTFChars(cache_dir_path, c_cache_dir_path);
}
#endif

MEDIAPIPE_API mp_packet* mp_create_packet_int(int value) {
    return new mp_packet {
        mediapipe::MakePacket<int>(value)
    };
}

MEDIAPIPE_API mp_packet* mp_create_packet_float(float value) {
    return new mp_packet {
        mediapipe::MakePacket<float>(value)
    };
}

MEDIAPIPE_API mp_packet* mp_create_packet_bool(bool value) {
    return new mp_packet {
        mediapipe::MakePacket<bool>(value)
    };
}

MEDIAPIPE_API mp_packet* mp_create_packet_image(mp_image image) {
    auto mp_frame = std::make_unique<mediapipe::ImageFrame>();
    auto mp_format = static_cast<mediapipe::ImageFormat::Format>(image.format);
    uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kDefaultAlignmentBoundary;
    mp_frame->CopyPixelData(mp_format, image.width, image.height, image.data, mp_alignment_boundary);

    return new mp_packet {
        mediapipe::Adopt(mp_frame.release())
    };
}

#if !MEDIAPIPE_DISABLE_GPU
MEDIAPIPE_API mp_packet* mp_create_packet_image_gpu(mp_instance* instance, mp_image image) {
    mp_packet *mp_pak;
    auto mp_frame = std::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGBA, 
        image.width, image.height, mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    auto mp_format = static_cast<mediapipe::ImageFormat::Format>(image.format);
    uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kDefaultAlignmentBoundary;
    mp_frame->CopyPixelData(mp_format, image.width, image.height, image.data, mp_alignment_boundary);
    absl::Status result = instance->gpu_helper.RunInGlContext([&instance, &mp_pak, &image, &mp_frame]() -> absl::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = instance->gpu_helper.CreateSourceTexture(*mp_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
        //   // Send GPU image packet into the graph.
        //   MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        //       kInputStream, mediapipe::Adopt(gpu_frame.release())
        //                         .At(mediapipe::Timestamp(frame_timestamp_us))));
          mp_pak = new mp_packet {
            mediapipe::Adopt(gpu_frame.release())
          };
          return absl::OkStatus();
        });
    return mp_pak;
}
#endif  // MEDIAPIPE_DISABLE_GPU

MEDIAPIPE_API mp_packet* mp_poll_packet(mp_poller* poller) {
    auto* packet = new mp_packet;
    poller->poller.Next(&packet->packet);
    return packet;
}

MEDIAPIPE_API void mp_destroy_packet(mp_packet* packet) {
    delete packet;
}

MEDIAPIPE_API const char* mp_get_packet_type(mp_packet* packet) {
    mediapipe::TypeId type = packet->packet.GetTypeId();
    std::string string = type.name();
    char* buffer = new char[string.size() + 1];
    std::strcpy(buffer, string.c_str());
    return buffer;
}

MEDIAPIPE_API void mp_free_packet_type(const char* type) {
    delete[] type;
}

MEDIAPIPE_API void mp_copy_packet_image(mp_packet* packet, uint8_t* out_data) {
    const auto& mp_frame = packet->packet.Get<mediapipe::ImageFrame>();
    size_t data_size = mp_frame.PixelDataSizeStoredContiguously();
    mp_frame.CopyToBuffer(out_data, data_size);
}

#if !MEDIAPIPE_DISABLE_GPU
MEDIAPIPE_API void mp_copy_packet_image_gpu(mp_instance* instance, mp_packet* packet, uint8_t* out_data) {
    std::unique_ptr<mediapipe::ImageFrame> mp_frame;
    absl::Status result = instance->gpu_helper.RunInGlContext([&instance, &packet, &out_data, &mp_frame]() -> absl::Status {
        auto& gpu_frame = packet->packet.Get<mediapipe::GpuBuffer>();
        auto texture = instance->gpu_helper.CreateSourceTexture(gpu_frame);
        mp_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
            gpu_frame.width(), gpu_frame.height(),
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        instance->gpu_helper.BindFramebuffer(texture);
        const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
            gpu_frame.format(), 0, instance->gpu_helper.GetGlVersion());
        glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                    info.gl_type, mp_frame->MutablePixelData());
        glFlush();
        texture.Release();
        size_t data_size = mp_frame->PixelDataSizeStoredContiguously();
        mp_frame->CopyToBuffer(out_data, data_size);
        return absl::OkStatus();
    });
    

}
#endif  // MEDIAPIPE_DISABLE_GPU

MEDIAPIPE_API mp_pose_landmark_list* mp_get_pose_landmarks(mp_packet* packet) {
    return get_pose_landmarks<mediapipe::LandmarkList, mediapipe::Landmark>(packet);
}

MEDIAPIPE_API mp_pose_landmark_list* mp_get_norm_pose_landmarks(mp_packet* packet) {
    return get_pose_landmarks<mediapipe::NormalizedLandmarkList, mediapipe::NormalizedLandmark>(packet);
}

MEDIAPIPE_API void mp_destroy_pose_landmarks(mp_pose_landmark_list* pose_landmarks) {
    for (int i = 0; i < pose_landmarks->length; i++) {
        delete[] pose_landmarks->elements[i].elements;
    }

    delete[] pose_landmarks->elements;
    delete pose_landmarks;
}

MEDIAPIPE_API mp_rect_list* mp_get_rects(mp_packet* packet) {
    return get_rects<mediapipe::Rect>(packet);
}

MEDIAPIPE_API mp_rect_list* mp_get_norm_rects(mp_packet* packet) {
    return get_rects<mediapipe::NormalizedRect>(packet);
}

MEDIAPIPE_API void mp_destroy_rects(mp_rect_list* list) {
    delete[] list->elements;
    delete list;
}

MEDIAPIPE_API const char* mp_get_last_error() {
    std::string string = last_error.ToString();
    char* buffer = new char[string.size() + 1];
    std::strcpy(buffer, string.c_str());
    return buffer;
}

MEDIAPIPE_API void mp_free_error(const char* message) {
    delete[] message;
}

}
