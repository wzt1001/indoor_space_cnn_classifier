import Augmentor

path_to_data = "./"

# Create a pipeline
p = Augmentor.Pipeline(path_to_data)

# Add some operations to an existing pipeline.

p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=200, height=200)

# Add a rotate90 operation to the pipeline:

# Here we sample 100,000 images from the pipeline.

# It is often useful to use scientific notation for specify
# large numbers with trailing zeros.
num_of_samples = int(1e5)

# Now we can sample from the pipeline:
p.sample(num_of_samples)

w, h = images[0].size

w_new = int(floor(w * self.percentage_area))
h_new = int(floor(h * self.percentage_area))

random_left_shift = random.randint(0, int((w - w_new)))  # Note: randint() is from uniform distribution.
random_down_shift = random.randint(0, int((h - h_new)))

def do(image):
	return image.crop((random_left_shift, random_down_shift, w_new + random_left_shift, h_new + random_down_shift))

augmented_images = []

for image in images:
	augmented_images.append(do(image))

return augmented_images