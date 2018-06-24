from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os

model_dir = '/work/isanchez/g/ds4-gdl-lrdecay/subpixel'
checkpoint_path = os.path.join(model_dir, "model-329")

print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False)