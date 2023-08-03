module attributes {torch.debug_module_name = "MaxPool"}{
	func.func @reduce_window_max_nhwc(%arg0: tensor<1x100x100x3xf32>) -> tensor<1x50x50x3xf32> {
		%0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
		%1 = "stablehlo.reduce_window" (%arg0, %0) ({
			^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
				%460 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
				stablehlo.return %460 : tensor<f32>
		}) {padding = dense<[[0,0], [0,0], [0,0], [0,0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x100x100x3xf32>, tensor<f32>) -> tensor<1x50x50x3xf32>
		return %1 : tensor<1x50x50x3xf32>
	}
	func.func @reduce_window_max_padding_nhwc(%arg0: tensor<1x100x100x3xf32>) -> tensor<1x51x51x3xf32> {
		%0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
			%1 = "stablehlo.reduce_window" (%arg0, %0) ({
					^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
					%460 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
					stablehlo.return %460 : tensor<f32>
					}) {padding = dense<[[0,0], [1,1], [1,1], [0,0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x100x100x3xf32>, tensor<f32>) -> tensor<1x51x51x3xf32>
		return %1 : tensor<1x51x51x3xf32>
	}
	func.func @reduce_window_max_nchw(%arg0: tensor<1x3x100x100xf32>) -> tensor<1x3x50x50xf32> {
		%0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
			%1 = "stablehlo.reduce_window" (%arg0, %0) ({
					^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
					%460 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
					stablehlo.return %460 : tensor<f32>
					}) {padding = dense<[[0,0], [0,0], [0,0], [0,0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x3x100x100xf32>, tensor<f32>) -> tensor<1x3x50x50xf32>
		return %1 : tensor<1x3x50x50xf32>
	}
	func.func @reduce_window_max_padding_nchw(%arg0: tensor<1x3x100x100xf32>) -> tensor<1x3x51x51xf32> {
		%0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
			%1 = "stablehlo.reduce_window" (%arg0, %0) ({
					^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
					%460 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
					stablehlo.return %460 : tensor<f32>
					}) {padding = dense<[[0,0], [0,0], [1,1], [1,1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x3x100x100xf32>, tensor<f32>) -> tensor<1x3x51x51xf32>
		return %1 : tensor<1x3x51x51xf32>
	}
}
		
