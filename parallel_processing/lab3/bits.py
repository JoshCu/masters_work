# Open pic1.bmp
# Loop over every pixel in the image
#  Get the RGB values (tuple)
#  Convert to grayscale (set R, G, B to average)
#  Set the pixel in the new image to the grayscale value
# Save new image

with open('pic1.bmp', 'rb') as file:
    data = file.read()

# Get the width and height from the image file
width = int.from_bytes(data[18:22], byteorder='little')
height = int.from_bytes(data[22:26], byteorder='little')

# Create a new bytearray to hold the image data
new_data = bytearray(data[:54])

pixel_list = []
# Loop over every pixel in the image
for row in range(height):
    for col in range(width):
        # Get the RGB values (tuple)
        index = row * width * 3 + col * 3 + 54
        b = data[index]
        g = data[index + 1]
        r = data[index + 2]

        # Convert to grayscale (set R, G, B to average)
        avg = (r + g + b) // 3
        pixel_list.append(avg)
        new_data += bytes([avg, avg, avg])

#write pixel list to file
with open('pixel_list.txt', 'w') as file:
    for pixel in pixel_list:
        file.write(str(pixel)+',')

# Save the new image
with open('pic1_grayscale.bmp', 'wb') as file:
    file.write(new_data)

