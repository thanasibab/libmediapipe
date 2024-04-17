package(default_visibility = [
    "//visibility:public",
])

cc_binary(
    name = "mediapipe",
    srcs = ["mediapipe.cpp", "mediapipe.h"],
    defines = ["MEDIAPIPE_EXPORT"],
    data = [
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_lite.tflite",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
        "//mediapipe/modules/pose_landmark:pose_landmark_gpu",

    ],
    linkshared = 1,
)