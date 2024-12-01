from PIL import Image

# Load the EPS file
eps_file = "Loss curve.eps"  # Replace with your EPS file path
image = Image.open(eps_file)

# Optionally convert to another mode (e.g., RGB) for better compatibility
image = image.convert("RGB")

# Display the image
image.show()

# Save it as another format (e.g., PNG)
image.save("Loss curve.png")
