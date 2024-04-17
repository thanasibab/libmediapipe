#!/bin/bash

set -e
shopt -s nullglob

VERSION="v0.8.11"
OPENCV_DIR=""
CONFIG="debug"

while [[ "$#" -gt 0 ]]; do
	case $1 in
		--version)
			VERSION="$2"
			shift
			shift
			;;
		--config)
			CONFIG="$2"
			shift
			shift
			;;
	esac
done

if [ "$CONFIG" != "debug" ] && [ "$CONFIG" != "release" ]; then
	echo "ERROR: Argument 'config' must be one of {debug, release}"
	exit 1
fi

echo "--------------------------------"
echo "CONFIGURATION"
echo "--------------------------------"

echo "MediaPipe version: $VERSION"
echo "Build configuration: $CONFIG"

OUTPUT_DIR="output"
PACKAGE_DIR="$OUTPUT_DIR/libmediapipe-$VERSION-aarch64-android"
DATA_DIR="$OUTPUT_DIR/data"

echo "--------------------------------"

set +e

echo -n "Checking Bazel (5.2.0) - "
BAZEL_BIN_PATH="$(type -P bazel-5.2.0)"
if [ -z "$BAZEL_BIN_PATH" ]; then
	echo "ERROR: Bazel (5.2.0) is not installed"
	exit 1
fi
echo "OK (Found at $BAZEL_BIN_PATH)"

echo -n "Checking Python - "
PYTHON_BIN_PATH="$(type -P python3)"
if [ -z "$PYTHON_BIN_PATH" ]; then
	echo "ERROR: Python is not installed"
	echo "Install Python with 'apt install python3'"
	exit 1
fi
echo "OK (Found at $PYTHON_BIN_PATH)"

set -e

echo "--------------------------------"
echo "CLONING MEDIAPIPE"
echo "--------------------------------"

if [ ! -d "mediapipe" ]; then
	git clone https://github.com/google/mediapipe.git
else#!/bin/bash

set -e
shopt -s nullglob

VERSION="v0.8.11"
OPENCV_DIR=""
CONFIG="debug"

while [[ "$#" -gt 0 ]]; do
	case $1 in
		--version)
			VERSION="$2"
			shift
			shift
			;;
		--config)
			CONFIG="$2"
			shift
			shift
			;;
	esac
done

if [ "$CONFIG" != "debug" ] && [ "$CONFIG" != "release" ]; then
	echo "ERROR: Argument 'config' must be one of {debug, release}"
	exit 1
fi

echo "--------------------------------"
echo "CONFIGURATION"
echo "--------------------------------"

echo "MediaPipe version: $VERSION"
echo "Build configuration: $CONFIG"

OUTPUT_DIR="output"
PACKAGE_DIR="$OUTPUT_DIR/libmediapipe-$VERSION-aarch64-android"
DATA_DIR="$OUTPUT_DIR/data"

echo "--------------------------------"

set +e

echo -n "Checking Bazel (6.1.1) - "
BAZEL_BIN_PATH="$(type -P bazel-6.1.1)"
if [ -z "$BAZEL_BIN_PATH" ]; then
	echo "ERROR: Bazel (6.1.1) is not installed"
	exit 1
fi
echo "OK (Found at $BAZEL_BIN_PATH)"

echo -n "Checking Python - "
PYTHON_BIN_PATH="$(type -P python3)"
if [ -z "$PYTHON_BIN_PATH" ]; then
	echo "ERROR: Python is not installed"
	echo "Install Python with 'apt install python3'"
	exit 1
fi
echo "OK (Found at $PYTHON_BIN_PATH)"

set -e

echo "--------------------------------"
echo "CLONING MEDIAPIPE"
echo "--------------------------------"

if [ ! -d "mediapipe" ]; then
	git clone https://github.com/google/mediapipe.git
else
	echo "Repository already cloned"
fi

cd mediapipe
git checkout "$VERSION"

echo -n "Setting up OpenCV - "
sed -i 's;3.4.3/opencv-3.4.3-android-sdk.zip;4.7.0/opencv-4.7.0-android-sdk.zip;g' WORKSPACE
sed -i 's;libopencv_java3;libopencv_java4;g' third_party/opencv_android.BUILD
echo "Done"

echo -n "Setting up the Android SDK - "
if ! grep -q "android_sdk_repository" WORKSPACE; then
	echo -e "\n\nandroid_sdk_repository(name=\"androidsdk\")" >> WORKSPACE
	echo "android_ndk_repository(name=\"androidndk\", api_level=21)" >> WORKSPACE
fi
echo "Done"

echo "--------------------------------"
echo "BUILDING C API"
echo "--------------------------------"

if [ -d "mediapipe/c" ]; then
	echo -n "Removing old C API - "
	rm -r mediapipe/c
	echo "Done"
fi

echo -n "Copying C API - "
cp -r ../c mediapipe/c
echo "Done"

if [ "$CONFIG" = "debug" ]; then
	BAZEL_CONFIG="dbg"
elif [ "$CONFIG" = "release" ]; then
	BAZEL_CONFIG="opt"
fi

bazel-6.1.1 build -c "$BAZEL_CONFIG" \
	--config=android_arm64 \
	--action_env PYTHON_BIN_PATH="$PYTHON_BIN_PATH" \
	--linkopt="-Wl,-soname,libmediapipe.so" \
	mediapipe/c:mediapipe

cd ..

if [ -d "$OUTPUT_DIR" ]; then
	echo -n "Removing existing output directory - "
	rm -rf "$OUTPUT_DIR"
	echo "Done"
fi

echo -n "Creating output directory - "
mkdir "$OUTPUT_DIR"
echo "Done"

echo -n "Creating library directories - "
mkdir "$PACKAGE_DIR"
mkdir "$PACKAGE_DIR/include"
mkdir "$PACKAGE_DIR/lib"
echo "Done"

echo -n "Copying libraries - "
cp mediapipe/bazel-bin/mediapipe/c/libmediapipe.so "$PACKAGE_DIR/lib"
cp $(find -L mediapipe/bazel-bin -name "libopencv_java4.so" | head -n 1) "$PACKAGE_DIR/lib"
echo "Done"

echo -n "Copying header - "
cp mediapipe/mediapipe/c/mediapipe.h "$PACKAGE_DIR/include"
echo "Done"

echo -n "Copying data - "

for DIR in mediapipe/bazel-bin/mediapipe/modules/*; do
	MODULE=$(basename "$DIR")
	mkdir -p "$DATA_DIR/mediapipe/modules/$MODULE"

	for FILE in "$DIR"/*.binarypb; do
		cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
	done

	for FILE in "$DIR"/*.tflite; do
		cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
	done
done

for DIR in mediapipe/bazel-bin/mediapipe/graphs/*; do
	GRAPH=$(basename "$DIR")
	mkdir -p "$DATA_DIR/mediapipe/graphs/$GRAPH"
	mkdir -p "$DATA_DIR/mediapipe/graphs/$GRAPH/subgraphs"

	for FILE in "$DIR"/*.binarypb; do
		cp "$FILE" "$DATA_DIR/mediapipe/graphs/$GRAPH/$(basename "$FILE")"
	done

	for FILE in "$DIR"/subgraphs/*.binarypb; do
		cp "$FILE" "$DATA_DIR/mediapipe/graphs/$GRAPH/subgraphs/$(basename "$FILE")"
	done

	for FILE in "$DIR"/*.tflite; do
		cp "$FILE" "$DATA_DIR/mediapipe/graphs/$GRAPH/$(basename "$FILE")"
	done
done

cp mediapipe/mediapipe/modules/hand_landmark/handedness.txt "$DATA_DIR/mediapipe/modules/hand_landmark"

echo "Done"

	echo "Repository already cloned"
fi

cd mediapipe
git checkout "$VERSION"

echo -n "Setting up OpenCV - "
sed -i 's;3.4.3/opencv-3.4.3-android-sdk.zip;4.5.0/opencv-4.5.0-android-sdk.zip;g' WORKSPACE
sed -i 's;libopencv_java3;libopencv_java4;g' third_party/opencv_android.BUILD
echo "Done"

echo -n "Setting up the Android SDK - "
if ! grep -q "android_sdk_repository" WORKSPACE; then
	echo -e "\n\nandroid_sdk_repository(name=\"androidsdk\")" >> WORKSPACE
	echo "android_ndk_repository(name=\"androidndk\", api_level=21)" >> WORKSPACE
fi
echo "Done"

echo "--------------------------------"
echo "BUILDING C API"
echo "--------------------------------"

if [ -d "mediapipe/c" ]; then
	echo -n "Removing old C API - "
	rm -r mediapipe/c
	echo "Done"
fi

echo -n "Copying C API - "
cp -r ../c mediapipe/c
echo "Done"

if [ "$CONFIG" = "debug" ]; then
	BAZEL_CONFIG="dbg"
elif [ "$CONFIG" = "release" ]; then
	BAZEL_CONFIG="opt"
fi

bazel-5.2.0 build -c "$BAZEL_CONFIG" \
	--config=android_arm64 \
	--action_env PYTHON_BIN_PATH="$PYTHON_BIN_PATH" \
	--linkopt="-Wl,-soname,libmediapipe.so" \
	mediapipe/c:mediapipe

cd ..

if [ -d "$OUTPUT_DIR" ]; then
	echo -n "Removing existing output directory - "
	rm -rf "$OUTPUT_DIR"
	echo "Done"
fi

echo -n "Creating output directory - "
mkdir "$OUTPUT_DIR"
echo "Done"

echo -n "Creating library directories - "
mkdir "$PACKAGE_DIR"
mkdir "$PACKAGE_DIR/include"
mkdir "$PACKAGE_DIR/lib"
echo "Done"

echo -n "Copying libraries - "
cp mediapipe/bazel-bin/mediapipe/c/libmediapipe.so "$PACKAGE_DIR/lib"
cp $(find -L mediapipe/bazel-bin -name "libopencv_java4.so" | head -n 1) "$PACKAGE_DIR/lib"
echo "Done"

echo -n "Copying header - "
cp mediapipe/mediapipe/c/mediapipe.h "$PACKAGE_DIR/include"
echo "Done"

echo -n "Copying data - "

for DIR in mediapipe/bazel-bin/mediapipe/modules/*; do
	MODULE=$(basename "$DIR")
	mkdir -p "$DATA_DIR/mediapipe/modules/$MODULE"

	for FILE in "$DIR"/*.binarypb; do
		cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
	done

	for FILE in "$DIR"/*.tflite; do
		cp "$FILE" "$DATA_DIR/mediapipe/modules/$MODULE/$(basename "$FILE")"
	done
done

for DIR in mediapipe/bazel-bin/mediapipe/graphs/*; do
	GRAPH=$(basename "$DIR")
	mkdir -p "$DATA_DIR/mediapipe/graphs/$GRAPH"
	mkdir -p "$DATA_DIR/mediapipe/graphs/$GRAPH/subgraphs"

	for FILE in "$DIR"/*.binarypb; do
		cp "$FILE" "$DATA_DIR/mediapipe/graphs/$GRAPH/$(basename "$FILE")"
	done

	for FILE in "$DIR"/subgraphs/*.binarypb; do
		cp "$FILE" "$DATA_DIR/mediapipe/graphs/$GRAPH/subgraphs/$(basename "$FILE")"
	done

	for FILE in "$DIR"/*.tflite; do
		cp "$FILE" "$DATA_DIR/mediapipe/graphs/$GRAPH/$(basename "$FILE")"
	done
done

cp mediapipe/mediapipe/modules/hand_landmark/handedness.txt "$DATA_DIR/mediapipe/modules/hand_landmark"

echo "Done"
