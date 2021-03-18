#include <cstdlib>

#include <glog/logging.h>
#include <absl/strings/string_view.h>

#pragma comment(lib, "allocation.lib")
#pragma comment(lib, "allocator.lib")
#pragma comment(lib, "api.lib")
#pragma comment(lib, "apply_multiplier.lib")
#pragma comment(lib, "arena_planner.lib")
#pragma comment(lib, "audio_utils.lib")
#pragma comment(lib, "avx2_ukernels.lib")
#pragma comment(lib, "avx512f_ukernels.lib")
#pragma comment(lib, "avx512skx_ukernels.lib")
#pragma comment(lib, "avx_ukernels.lib")
#pragma comment(lib, "bad_optional_access.lib")
#pragma comment(lib, "bad_variant_access.lib")
#pragma comment(lib, "base.lib")
#pragma comment(lib, "blocking_counter.lib")
#pragma comment(lib, "block_map.lib")
#pragma comment(lib, "builtin_ops.lib")
#pragma comment(lib, "builtin_op_kernels.lib")
#pragma comment(lib, "city.lib")
#pragma comment(lib, "civil_time.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "common.lo.lib")
#pragma comment(lib, "config.lib")
#pragma comment(lib, "context.lib")
#pragma comment(lib, "context_get_ctx.lib")
#pragma comment(lib, "cord.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "cpuinfo_impl.lib")
#pragma comment(lib, "cpu_backend_context.lib")
#pragma comment(lib, "cpu_backend_gemm.lib")
#pragma comment(lib, "cpu_check.lib")
#pragma comment(lib, "ctx.lib")
#pragma comment(lib, "debugging_internal.lib")
#pragma comment(lib, "debug_log.lib")
#pragma comment(lib, "demangle_internal.lib")
#pragma comment(lib, "dynamic_annotations.lib")
#pragma comment(lib, "eigen_support.lib")
#pragma comment(lib, "error_reporter.lib")
#pragma comment(lib, "exponential_biased.lib")
#pragma comment(lib, "external_cpu_backend_context.lib")
#pragma comment(lib, "farmhash.lib")
#pragma comment(lib, "fft2d.lib")
#pragma comment(lib, "flag.lib")
#pragma comment(lib, "flag_internal.lib")
#pragma comment(lib, "flatbuffers.lib")
#pragma comment(lib, "fma3_ukernels.lib")
#pragma comment(lib, "format_converter.lib")
#pragma comment(lib, "framework_lib.lo.lib")
#pragma comment(lib, "frontend.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "graphcycles_internal.lib")
#pragma comment(lib, "hash.lib")
#pragma comment(lib, "hashtablez_sampler.lib")
#pragma comment(lib, "have_built_path_for_avx2_fma.lib")
#pragma comment(lib, "have_built_path_for_avx512.lib")
#pragma comment(lib, "indirection.lib")
#pragma comment(lib, "instrumentation.lib")
#pragma comment(lib, "int128.lib")
#pragma comment(lib, "internal.lib")
#pragma comment(lib, "kernel_arm.lib")
#pragma comment(lib, "kernel_avx2_fma.lib")
#pragma comment(lib, "kernel_avx512.lib")
#pragma comment(lib, "kernel_util.lib")
#pragma comment(lib, "kernel_utils.lib")
#pragma comment(lib, "leak_check.lib")
#pragma comment(lib, "log_severity.lib")
#pragma comment(lib, "lstm_eval.lib")
#pragma comment(lib, "malloc_internal.lib")
#pragma comment(lib, "marshalling.lib")
#pragma comment(lib, "memory_planner.lib")
#pragma comment(lib, "minimal_logging.lib")
#pragma comment(lib, "mutable_op_resolver.lib")
#pragma comment(lib, "neon_tensor_utils.lib")
#pragma comment(lib, "operators.lib")
#pragma comment(lib, "operator_run.lib")
#pragma comment(lib, "op_resolver.lib")
#pragma comment(lib, "packing.lib")
#pragma comment(lib, "pack_arm.lib")
#pragma comment(lib, "pack_avx2_fma.lib")
#pragma comment(lib, "pack_avx512.lib")
#pragma comment(lib, "platform_profiler.lib")
#pragma comment(lib, "portable_tensor_utils.lib")
#pragma comment(lib, "prepacked_cache.lib")
#pragma comment(lib, "prepare_packed_matrices.lib")
#pragma comment(lib, "program_name.lib")
#pragma comment(lib, "protobuf.lo.lib")
#pragma comment(lib, "protobuf_lite.lo.lib")
#pragma comment(lib, "pthreadpool.lib")
#pragma comment(lib, "quantization_util.lib")
#pragma comment(lib, "raw_hash_set.lib")
#pragma comment(lib, "raw_logging_internal.lib")
#pragma comment(lib, "registry.lib")
#pragma comment(lib, "resource.lib")
#pragma comment(lib, "scalar_ukernels.lib")
#pragma comment(lib, "schema_utils.lib")
#pragma comment(lib, "simple_memory_arena.lib")
#pragma comment(lib, "spinlock_wait.lib")
#pragma comment(lib, "sse2_ukernels.lib")
#pragma comment(lib, "sse41_ukernels.lib")
#pragma comment(lib, "sse_tensor_utils.lib")
#pragma comment(lib, "ssse3_ukernels.lib")
#pragma comment(lib, "stacktrace.lib")
#pragma comment(lib, "status.lib")
#pragma comment(lib, "stderr_reporter.lib")
#pragma comment(lib, "strings.lib")
#pragma comment(lib, "string_util.lib")
#pragma comment(lib, "str_format_internal.lib")
#pragma comment(lib, "symbolize.lib")
#pragma comment(lib, "synchronization.lib")
#pragma comment(lib, "system_aligned_alloc.lib")
#pragma comment(lib, "tables.lib")
#pragma comment(lib, "tensor_utils.lib")
#pragma comment(lib, "tflite_with_xnnpack_optional.lib")
#pragma comment(lib, "thread_pool.lib")
#pragma comment(lib, "throw_delegate.lib")
#pragma comment(lib, "time.lib")
#pragma comment(lib, "time_zone.lib")
#pragma comment(lib, "transpose_utils.lib")
#pragma comment(lib, "trmul.lib")
#pragma comment(lib, "tune.lib")
#pragma comment(lib, "util.lib")
#pragma comment(lib, "version_info.lib")
#pragma comment(lib, "wait.lib")
#pragma comment(lib, "xnnpack_delegate.lib")
#pragma comment(lib, "xnnpack_f32.lib")
#pragma comment(lib, "xop_ukernels.lib")


#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"

#pragma comment(lib, "anchor_cc_proto.lo.lib")
#pragma comment(lib, "annotation_overlay_calculator.lo.lib")
#pragma comment(lib, "annotation_overlay_calculator_cc_proto.lo.lib")
#pragma comment(lib, "annotation_renderer.lib")
#pragma comment(lib, "association_calculator_cc_proto.lo.lib")
#pragma comment(lib, "association_norm_rect_calculator.lo.lib")
#pragma comment(lib, "begin_loop_calculator.lo.lib")
#pragma comment(lib, "calculator_base.lib")
#pragma comment(lib, "calculator_cc_proto.lo.lib")
#pragma comment(lib, "calculator_context.lib")
#pragma comment(lib, "calculator_context_manager.lib")
#pragma comment(lib, "calculator_contract.lib")
#pragma comment(lib, "calculator_graph.lib")
#pragma comment(lib, "calculator_graph_template_cc_proto.lo.lib")
#pragma comment(lib, "calculator_node.lib")
#pragma comment(lib, "calculator_options_cc_proto.lo.lib")
#pragma comment(lib, "calculator_profile_cc_proto.lo.lib")
#pragma comment(lib, "calculator_registry_util.lib")
#pragma comment(lib, "calculator_state.lib")
#pragma comment(lib, "callback_packet_calculator.lo.lib")
#pragma comment(lib, "callback_packet_calculator_cc_proto.lo.lib")
#pragma comment(lib, "classification_cc_proto.lo.lib")
#pragma comment(lib, "clip_vector_size_calculator.lo.lib")
#pragma comment(lib, "clip_vector_size_calculator_cc_proto.lo.lib")
#pragma comment(lib, "clock.lib")
#pragma comment(lib, "collection_has_min_size_calculator.lo.lib")
#pragma comment(lib, "collection_has_min_size_calculator_cc_proto.lo.lib")
#pragma comment(lib, "collection_item_id.lib")
#pragma comment(lib, "color_cc_proto.lo.lib")
#pragma comment(lib, "constant_side_packet_calculator.lo.lib")
#pragma comment(lib, "constant_side_packet_calculator_cc_proto.lo.lib")
#pragma comment(lib, "counter_factory.lib")
#pragma comment(lib, "cpu_op_resolver.lo.lib")
#pragma comment(lib, "cpu_util.lib")
#pragma comment(lib, "default_input_stream_handler.lo.lib")
#pragma comment(lib, "default_input_stream_handler_cc_proto.lo.lib")
#pragma comment(lib, "delegating_executor.lib")
#pragma comment(lib, "detections_to_rects_calculator.lo.lib")
#pragma comment(lib, "detections_to_rects_calculator_cc_proto.lo.lib")
#pragma comment(lib, "detections_to_render_data_calculator.lo.lib")
#pragma comment(lib, "detections_to_render_data_calculator_cc_proto.lo.lib")
#pragma comment(lib, "detection_cc_proto.lo.lib")
#pragma comment(lib, "detection_letterbox_removal_calculator.lo.lib")
#pragma comment(lib, "end_loop_calculator.lo.lib")
#pragma comment(lib, "executor.lib")
#pragma comment(lib, "file_helpers.lib")
#pragma comment(lib, "file_path.lib")
#pragma comment(lib, "fill_packet_set.lib")
#pragma comment(lib, "filter_collection_calculator.lo.lib")
#pragma comment(lib, "fixed_size_input_stream_handler.lo.lib")
#pragma comment(lib, "fixed_size_input_stream_handler_cc_proto.lo.lib")
#pragma comment(lib, "flow_limiter_calculator.lo.lib")
#pragma comment(lib, "gate_calculator.lo.lib")
#pragma comment(lib, "gate_calculator_cc_proto.lo.lib")
#pragma comment(lib, "gpu_service.lib")
#pragma comment(lib, "graph_output_stream.lib")
#pragma comment(lib, "graph_profiler_real.lib")
#pragma comment(lib, "graph_tracer.lib")
#pragma comment(lib, "hand_landmarks_to_rect_calculator.lo.lib")
#pragma comment(lib, "hand_landmark_cpu.lo.lib")
#pragma comment(lib, "hand_landmark_landmarks_to_roi.lo.lib")
#pragma comment(lib, "hand_landmark_tracking_cpu.lo.lib")
#pragma comment(lib, "hand_renderer_cpu.lo.lib")
#pragma comment(lib, "header_util.lib")
#pragma comment(lib, "image_format_cc_proto.lo.lib")
#pragma comment(lib, "image_frame.lib")
#pragma comment(lib, "image_frame_opencv.lib")
#pragma comment(lib, "image_properties_calculator.lo.lib")
#pragma comment(lib, "image_to_tensor_calculator.lo.lib")
#pragma comment(lib, "image_to_tensor_calculator_cc_proto.lo.lib")
#pragma comment(lib, "image_to_tensor_converter_opencv.lib")
#pragma comment(lib, "image_to_tensor_utils.lib")
#pragma comment(lib, "immediate_input_stream_handler.lo.lib")
#pragma comment(lib, "immediate_mux_calculator.lo.lib")
#pragma comment(lib, "inference_calculator.lo.lib")
#pragma comment(lib, "inference_calculator_cc_proto.lo.lib")
#pragma comment(lib, "input_side_packet_handler.lib")
#pragma comment(lib, "input_stream_handler.lib")
#pragma comment(lib, "input_stream_manager.lib")
#pragma comment(lib, "input_stream_shard.lib")
#pragma comment(lib, "in_order_output_stream_handler.lo.lib")
#pragma comment(lib, "labels_to_render_data_calculator.lo.lib")
#pragma comment(lib, "labels_to_render_data_calculator_cc_proto.lo.lib")
#pragma comment(lib, "landmarks_to_render_data_calculator.lo.lib")
#pragma comment(lib, "landmarks_to_render_data_calculator_cc_proto.lo.lib")
#pragma comment(lib, "landmark_cc_proto.lo.lib")
#pragma comment(lib, "landmark_letterbox_removal_calculator.lo.lib")
#pragma comment(lib, "landmark_projection_calculator.lo.lib")
#pragma comment(lib, "landmark_projection_calculator_cc_proto.lo.lib")
#pragma comment(lib, "legacy_calculator_support.lib")
#pragma comment(lib, "location.lo.lib")
#pragma comment(lib, "location_data_cc_proto.lo.lib")
#pragma comment(lib, "locus_cc_proto.lo.lib")
#pragma comment(lib, "logic_calculator.lo.lib")
#pragma comment(lib, "logic_calculator_cc_proto.lo.lib")
#pragma comment(lib, "matrix.lib")
#pragma comment(lib, "matrix_data_cc_proto.lo.lib")
#pragma comment(lib, "max_pool_argmax.lib")
#pragma comment(lib, "max_unpooling.lib")
#pragma comment(lib, "mediapipe_options_cc_proto.lo.lib")
#pragma comment(lib, "merge_calculator.lo.lib")
#pragma comment(lib, "name_util.lib")
#pragma comment(lib, "non_max_suppression_calculator.lo.lib")
#pragma comment(lib, "non_max_suppression_calculator_cc_proto.lo.lib")
#pragma comment(lib, "opencv_video_decoder_calculator.lo.lib")
#pragma comment(lib, "opencv_video_encoder_calculator.lo.lib")
#pragma comment(lib, "opencv_video_encoder_calculator_cc_proto.lo.lib")
#pragma comment(lib, "op_resolver_mediapipe.lib")
#pragma comment(lib, "output_side_packet_impl.lib")
#pragma comment(lib, "output_stream_handler.lib")
#pragma comment(lib, "output_stream_manager.lib")
#pragma comment(lib, "output_stream_shard.lib")
#pragma comment(lib, "packet.lib")
#pragma comment(lib, "packet_factory_cc_proto.lo.lib")
#pragma comment(lib, "packet_generator_cc_proto.lo.lib")
#pragma comment(lib, "packet_generator_graph.lib")
#pragma comment(lib, "packet_inner_join_calculator.lo.lib")
#pragma comment(lib, "packet_type.lib")
#pragma comment(lib, "palm_detection_cpu.lo.lib")
#pragma comment(lib, "palm_detection_detection_to_roi.lo.lib")
#pragma comment(lib, "palm_detection_gpu.lo.lib")
#pragma comment(lib, "pass_through_calculator.lo.lib")
#pragma comment(lib, "previous_loopback_calculator.lo.lib")
#pragma comment(lib, "profiler_resource_util.lib")
#pragma comment(lib, "proto_descriptor_cc_proto.lo.lib")
#pragma comment(lib, "proto_util_lite.lib")
#pragma comment(lib, "rasterization_cc_proto.lo.lib")
#pragma comment(lib, "rect_cc_proto.lo.lib")
#pragma comment(lib, "rect_to_render_data_calculator.lo.lib")
#pragma comment(lib, "rect_to_render_data_calculator_cc_proto.lo.lib")
#pragma comment(lib, "rect_transformation_calculator.lo.lib")
#pragma comment(lib, "rect_transformation_calculator_cc_proto.lo.lib")
#pragma comment(lib, "registration.lib")
#pragma comment(lib, "registration_token.lib")
#pragma comment(lib, "render_data_cc_proto.lo.lib")
#pragma comment(lib, "resource_util.lib")
#pragma comment(lib, "ret_check.lib")
#pragma comment(lib, "scheduler_queue.lib")
#pragma comment(lib, "sink.lo.lib")
#pragma comment(lib, "split_normalized_landmark_list_calculator.lo.lib")
#pragma comment(lib, "split_vector_calculator.lo.lib")
#pragma comment(lib, "split_vector_calculator_cc_proto.lo.lib")
#pragma comment(lib, "ssd_anchors_calculator.lo.lib")
#pragma comment(lib, "ssd_anchors_calculator_cc_proto.lo.lib")
#pragma comment(lib, "status_mediapipe.lib")
#pragma comment(lib, "statusor.lib")
#pragma comment(lib, "status_handler_cc_proto.lo.lib")
#pragma comment(lib, "status_util.lib")
#pragma comment(lib, "stream_handler_cc_proto.lo.lib")
#pragma comment(lib, "subgraph.lib")
#pragma comment(lib, "subgraph_expansion.lib")
#pragma comment(lib, "tag_map.lib")
#pragma comment(lib, "template_expander.lib")
#pragma comment(lib, "tensor.lib")
#pragma comment(lib, "tensors_to_classification_calculator.lo.lib")
#pragma comment(lib, "tensors_to_classification_calculator_cc_proto.lo.lib")
#pragma comment(lib, "tensors_to_detections_calculator.lo.lib")
#pragma comment(lib, "tensors_to_detections_calculator_cc_proto.lo.lib")
#pragma comment(lib, "tensors_to_floats_calculator.lo.lib")
#pragma comment(lib, "tensors_to_landmarks_calculator.lo.lib")
#pragma comment(lib, "tensors_to_landmarks_calculator_cc_proto.lo.lib")
#pragma comment(lib, "tflite_custom_op_resolver_calculator.lo.lib")
#pragma comment(lib, "tflite_custom_op_resolver_calculator_cc_proto.lo.lib")
#pragma comment(lib, "threadpool.lib")
#pragma comment(lib, "thread_pool_executor.lib")
#pragma comment(lib, "thread_pool_executor_cc_proto.lo.lib")
#pragma comment(lib, "thresholding_calculator.lo.lib")
#pragma comment(lib, "thresholding_calculator_cc_proto.lo.lib")
#pragma comment(lib, "timestamp.lib")
#pragma comment(lib, "topologicalsorter.lib")
#pragma comment(lib, "transpose_conv_bias.lib")
#pragma comment(lib, "validate.lib")
#pragma comment(lib, "validated_graph_config.lib")
#pragma comment(lib, "validate_name.lib")


#include "libhandlandmarks.h"
#pragma comment(lib, "libhandlandmarks.lib")


#ifdef _DEBUG
#pragma comment(lib,"opencv_calib3d440d.lib")
#pragma comment(lib,"opencv_core440d.lib")
#pragma comment(lib,"opencv_features2d440d.lib")
#pragma comment(lib,"opencv_flann440d.lib")
#pragma comment(lib,"opencv_highgui440d.lib")
#pragma comment(lib,"opencv_imgcodecs440d.lib")
#pragma comment(lib,"opencv_imgproc440d.lib")
#pragma comment(lib,"opencv_objdetect440d.lib")
#pragma comment(lib,"opencv_videoio440d.lib")
#pragma comment(lib,"opencv_video440d.lib")
#pragma comment(lib,"opencv_ml440d.lib")
#else
#pragma comment(lib,"opencv_calib3d440.lib")
#pragma comment(lib,"opencv_core440.lib")
#pragma comment(lib,"opencv_features2d440.lib")
#pragma comment(lib,"opencv_flann440.lib")
#pragma comment(lib,"opencv_highgui440.lib")
#pragma comment(lib,"opencv_imgcodecs440.lib")
#pragma comment(lib,"opencv_imgproc440.lib")
#pragma comment(lib,"opencv_objdetect440.lib")
#pragma comment(lib,"opencv_videoio440.lib")
#pragma comment(lib,"opencv_video440.lib")
#pragma comment(lib,"opencv_ml440.lib")
#endif

int main()
{
	google::InitGoogleLogging("");
	LOG(INFO) << "glog is initialized";

	//std::string s = "test abseil string_view";
	//absl::string_view sv(s);
	//std::cout << sv << std::endl;

	std::string calculator_graph_config_contents = R"(
input_stream: "input_video"
output_stream: "output_video"
output_stream: "landmarks"

node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:num_hands"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: 1 }
    }
  }
}

node {
  calculator: "HandLandmarkTrackingCpu"
  input_stream: "IMAGE:input_video"
  input_side_packet: "NUM_HANDS:num_hands"
  output_stream: "LANDMARKS:landmarks"
  output_stream: "HANDEDNESS:handedness"
  output_stream: "PALM_DETECTIONS:multi_palm_detections"
  output_stream: "HAND_ROIS_FROM_LANDMARKS:multi_hand_rects"
  output_stream: "HAND_ROIS_FROM_PALM_DETECTIONS:multi_palm_rects"
}
)";

	::mediapipe::StatusOrPoller statuspoller = InitGraph(calculator_graph_config_contents);
	const ::mediapipe::Status &init_status = statuspoller.status();
	if (!init_status.ok())
	{
		LOG(ERROR) << "Failed to run the graph: " << init_status.message();
		return -1;
	}
	LOG(INFO) << "Succeeded to run the graph: " << init_status.message();
	::mediapipe::OutputStreamPoller &poller = statuspoller.ValueOrDie();

	RunGraph();

	cv::VideoCapture capture;
	capture.open(0);

	bool grab_frames = true;
	while (grab_frames)
	{
		cv::Mat camera_frame_raw;
		capture >> camera_frame_raw;
		if (camera_frame_raw.empty()) break;
		cv::Mat camera_frame;
		cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
		cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
		std::vector<mediapipe::NormalizedLandmarkList> vec_landmarks;
		if (!ProcessFrame(poller, camera_frame, vec_landmarks).ok())
			break;

        for (int i = 0; i < vec_landmarks.size(); ++i)
        {
            auto& landmarks = vec_landmarks[i];
            if (landmarks.landmark_size() < 21)
                LOG(INFO) << "num " << landmarks.landmark_size();
            else
            {
                const mediapipe::NormalizedLandmark& landmark4 = landmarks.landmark(4);
                const mediapipe::NormalizedLandmark& landmark3 = landmarks.landmark(3);
                const mediapipe::NormalizedLandmark& landmark2 = landmarks.landmark(2);
                const mediapipe::NormalizedLandmark& landmark8 = landmarks.landmark(8);
                const mediapipe::NormalizedLandmark& landmark7 = landmarks.landmark(7);
                const mediapipe::NormalizedLandmark& landmark6 = landmarks.landmark(6);
                const mediapipe::NormalizedLandmark& landmark12 = landmarks.landmark(12);
                const mediapipe::NormalizedLandmark& landmark11 = landmarks.landmark(11);
                const mediapipe::NormalizedLandmark& landmark10 = landmarks.landmark(10);
                const mediapipe::NormalizedLandmark& landmark16 = landmarks.landmark(16);
                const mediapipe::NormalizedLandmark& landmark15 = landmarks.landmark(15);
                const mediapipe::NormalizedLandmark& landmark14 = landmarks.landmark(14);
                const mediapipe::NormalizedLandmark& landmark20 = landmarks.landmark(20);
                const mediapipe::NormalizedLandmark& landmark19 = landmarks.landmark(19);
                const mediapipe::NormalizedLandmark& landmark18 = landmarks.landmark(18);
                if (landmark8.y() < landmark7.y() && landmark7.y() < landmark6.y() &&
                    landmark12.y() > landmark11.y() &&
                    landmark16.y() > landmark15.y() &&
                    landmark20.y() > landmark19.y())
                {
                    LOG(INFO) << "number 1";
                }
                if (landmark8.y() < landmark7.y() && landmark7.y() < landmark6.y() &&
                    landmark12.y() < landmark11.y() && landmark11.y() < landmark10.y() &&
                    landmark16.y() > landmark15.y() &&
                    landmark20.y() > landmark19.y())
                {
                    LOG(INFO) << "number 2";
                }
                if (landmark8.y() > landmark7.y() &&
                    landmark12.y() < landmark11.y() && landmark11.y() < landmark10.y() &&
                    landmark16.y() < landmark15.y() && landmark15.y() < landmark14.y() &&
                    landmark20.y() < landmark19.y() && landmark19.y() < landmark18.y())
                {
                    LOG(INFO) << "number 3";
                }
                if (((landmark4.x() > landmark8.x() && landmark4.x() < landmark20.x()) || (landmark4.x() < landmark8.x() && landmark4.x() > landmark20.x())) &&
                    landmark8.y() < landmark7.y() && landmark7.y() < landmark6.y() &&
                    landmark12.y() < landmark11.y() && landmark11.y() < landmark10.y() &&
                    landmark16.y() < landmark15.y() && landmark15.y() < landmark14.y() &&
                    landmark20.y() < landmark19.y() && landmark19.y() < landmark18.y())
                {
                    LOG(INFO) << "number 4";
                }
                if (((landmark4.x() > landmark8.x() && landmark4.x() > landmark20.x()) || (landmark4.x() < landmark8.x() && landmark4.x() < landmark20.x())) &&
                    landmark4.y() < landmark3.y() && landmark3.y() < landmark2.y() &&
                    landmark8.y() < landmark7.y() && landmark7.y() < landmark6.y() &&
                    landmark12.y() < landmark11.y() && landmark11.y() < landmark10.y() &&
                    landmark16.y() < landmark15.y() && landmark15.y() < landmark14.y() &&
                    landmark20.y() < landmark19.y() && landmark19.y() < landmark18.y())
                {
                    LOG(INFO) << "number 5";
                }
            }
        }

		const int pressed_key = cv::waitKey(5);
		if (pressed_key >= 0 && pressed_key != 255)
			grab_frames = false;
	}

    LOG(INFO) << "Shutting down.";
	CloseGraph();

	return 0;
}