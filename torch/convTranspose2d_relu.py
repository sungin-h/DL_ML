import torch
import torch_mlir

class NN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.upsample_relu = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(3,3,2,2,0),
			torch.nn.ReLU(),
		)
	def forward(self, x):
		a = self.upsample_relu(x)
		return a

model = NN().eval()

#print(model)
input_tensor = torch.randn(1,3,32,64)
output_tensor = model(input_tensor)
print("input tensor shapes:",input_tensor.shape)
print("output tensor shapes:",output_tensor.shape)


scripted = torch.jit.script(model,input_tensor)
scripted_output = scripted(input_tensor)
print("scripted output tensor shapes:",scripted_output.shape)


module = torch_mlir.compile(model,input_tensor,output_type=torch_mlir.OutputType.TOSA)

with open("convtr_re_concat.tosa","w") as f:
	f.write(str(module))
