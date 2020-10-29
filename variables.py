import os 
train_dir = os.path.join(os.getcwd(), 'data/Train/')
test_dir = os.path.join(os.getcwd(), 'data/Test/')
model_weights = "data/weights/doggy_mobilenetW1.h5"
model_architecture = "data/weights/doggy_mobilenetW1.json"

test_split = 0.2
val_split = 0.15
seed = 42

target_size=(224, 224)
batch_size = 8
valid_size = 4
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.2
zoom_range = 0.15
rotation_range = 20
shift_range = 0.2
rescale = 1./255
dense_1 = 512
dense_2 = 256
dense_3 = 64
epochs = 10
verbose = 1