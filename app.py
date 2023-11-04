#pip install pydot
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import torchvision
import os
import glob
from tensorflow.keras import layers, models
import torch.nn as nn
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define a dictionary to keep track of the selected tab
if "selected_tab" not in st.session_state:
    print(st.session_state)
    st.session_state.selected_tab= "Translate file"
    print(st.session_state.selected_tab)


with st.sidebar:
    st.title('🧠✍️📚 Hand written text classification Project')
    tabs = st.radio("Select Dataset Maker Tab:", ["Home page", "Prepare datset", "Create Neural net","Training",'Inference'], key='tabs', index=0)

# Update the selected tab
st.session_state.selected_tab = tabs

# Get the active tab name
tab_name = st.session_state.selected_tab

st.subheader(tabs)

if tab_name == 'Home page':
    st.title("# Handwritten Digit Recognition")
    st.header('architecture diagram')
    st.image('no code gui.png')




Network= torch.load('model_torch_MNIST_CNN_99_1_streamlit.chk')



if tab_name == 'Prepare datset':
    st.header('Creat Dataset or upload')
    # Specify canvas parameters in application
    stroke_width1 = st.sidebar.slider("Stroke width1: ", 1, 25, 9)
    # Create a canvas component
    st.write("Create a dataset for handwritten classification.")
    class_counts = {}  

    # Number of classes
    num_classes = st.number_input("Number of Classes", min_value=1, value=3)
  
    class_names = st.text_input('Enter class names separated by commas').split(',')

# Create a dropdown button to show/hide canvas and upload button
#is_dropdown_expanded = st.checkbox(f"Class {i + 1} Details", key=f"dropdown_{i}")

#if is_dropdown_expanded:
    # Create a canvas for drawing
    canvas = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width1,
    stroke_color='#FFFFFF',
    background_color='#000000',
    #background_image=Image.open(bg_image) if bg_image else None,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas_class",
)
    selected_class = st.selectbox('Select class', class_names)
    # Create a button to save the canvas drawing
    if st.button(f"Save image for Class {selected_class}"):
        # Add your code to save the canvas drawing and uploaded image
        if canvas.image_data is not None:
            # Get the numpy array (4-channel RGBA 100,100,4)
                input_numpy_array = np.array(canvas.image_data)
                
                
                # Get the RGBA PIL image
                input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')                        
                # Convert it to grayscale
                input_image_gs = input_image.convert('L')
                input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200,200)
                # st.write('### Image as a grayscale Numpy array')
                # st.write(input_image_gs_np)
                
                # Create a temporary image for opencv to read it
                input_image_gs.save('temp_for_cv2.jpg')
                image = Image.open('temp_for_cv2.jpg', 0)
                    
                if os.path.exists('temp_for_cv2.jpg'):
                        # Delete the file
                        os.remove('temp_for_cv2.jpg')
                # Start creating a bounding box
                width, height = image.size

                # Create a drawing context
                draw = ImageDraw.Draw(image)

                x, y, w, h = (0, 0, width, height)
                # Create new blank image and shift ROI to new coordinates
                ROI = image[y:y+h, x:x+w]
                mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
                width, height = mask.shape
            #     print(ROI.shape)
            #     print(mask.shape)
                x = width//2 - ROI.shape[0]//2 
                y = height//2 - ROI.shape[1]//2 
            #     print(x,y)
                mask[y:y+h, x:x+w] = ROI
            #     print(mask)
                # Check if centering/masking was successful
            #     plt.imshow(mask, cmap='viridis') 
                output_image = Image.fromarray(mask) # mask has values in [0-255] as expected
                # Now we need to resize, but it causes problems with default arguments as it changes the range of pixel values to be negative or positive
                # compressed_output_image = output_image.resize((22,22))
                # Therefore, we use the following:
                compressed_output_image = output_image.resize((22,22), Image.BILINEAR) # PIL.Image.NEAREST or PIL.Image.BILINEAR also performs good

                convert_tensor = torchvision.transforms.ToTensor()
                tensor_image = convert_tensor(compressed_output_image)
                # Another problem we face is that in the above ToTensor() command, we should have gotten a normalized tensor with pixel values in [0,1]
                # But somehow it doesn't happen. Therefore, we need to normalize manually
                tensor_image = tensor_image/255.
                # Padding
                tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
                # Normalization shoudl be done after padding i guess
                convert_tensor = torchvision.transforms.Normalize((0.1307), (0.3081)) # Mean and std of MNIST
                tensor_image = convert_tensor(tensor_image)

                # The following gives noisy image because the values are from -1 to 1, which is not a proper image format
                im = Image.fromarray(tensor_image.detach().cpu().numpy().reshape(28,28), mode='L')
                #im.save("processed_tensor.png", "PNG")
                # So we use matplotlib to save it instead
                if selected_class not in class_counts:
                    class_counts[selected_class] = 1
                else:
                    class_counts[selected_class] += 1
                count = class_counts[selected_class]
                img_paths = glob.glob(f'dataset/{selected_class}_*.png')
                counts = [int(p.split('_')[-1].split('.')[0]) for p in img_paths]
                                
                if counts:
                    # Start count from max existing file
                    count = max(counts) + 1 
                else:
                    # Start from 1 if no existing images
                    count = 1

                plt.imsave(f"dataset/{selected_class}_{count}.png",tensor_image.detach().cpu().numpy().reshape(28,28), cmap='gray')

        else:
                st.write('Please draw letters in canvas')




if tab_name == 'Create Neural net':

    
    #st.iframe(url, width=700, height=600)
    st.title("How cnn works")

    st.header('We will build 3 layer of convolution layer and 2 dense layer of cnn network for predicting handwritten')
    
    st.text("Convolution (Conv2D): Think of convolution as a way for the computer to look at different parts of an image and find important patterns.")
               
    st.text("Max Pooling (MaxPooling2D): Max pooling is like zooming out. It helps the computer see the bigger picture and understand the main shapes and objects.")

    st.text("Dropout : Dropout is like taking a short break to learn better. It helps prevent getting too good at one thing and missing out on others.")
    
    st.text("Flatten : Flattening is like making all the puzzle pieces into a straight line, so we can use them to make decisions.")
    
    st.text("Dense (Fully Connected) Layers : Imagine the computer has a lot of small boxes to put different things in. Each box is like a different idea or category, and the computer uses them to make decisions.")
    
    st.text("Activation Functions: Activation functions are like filters for information. They help the computer see important things and make sense of what it's looking at.")

    st.text("Optimizer: The optimizer is like a teacher that helps the computer get better at recognizing things. It guides the computer's learning process.")
    
    st.header('for better explanation and understand of each concept of neural net we can create playground like this')

    url = 'https://poloclub.github.io/cnn-explainer/'

    html = f'<iframe src="https://poloclub.github.io/cnn-explainer/" height="900" width="100%"></iframe>'

    st.components.v1.html(html, height=1000,width=900)

    st.title("CNN Builder")
    # User inputs


   
    dropout_rate = st.slider("Dropout rate:", 0.0, 0.5, 0.5)
    activation = st.selectbox('Activation', ['relu', 'tanh'])
    kernel_size = st.slider('Kernel size', 2, 4, 2) 
    pool_size = st.slider('Pool size', 2, 4, 2)
    num_output_classes = st.number_input("Number of output classes:", min_value=1, value=10)

   
    
    st.session_state.model = Sequential()
    st.session_state.model.add(Conv2D(64,(kernel_size,kernel_size),input_shape=(28,28,4),activation=activation,padding='same'))
    st.session_state.model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
    st.session_state.model.add(Conv2D(64,(kernel_size,kernel_size),input_shape=(28,28,4),activation=activation,padding='same'))
    st.session_state.model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
    st.session_state.model.add(Dropout(dropout_rate))
    st.session_state.model.add(Conv2D(128,(kernel_size,kernel_size),input_shape=(28,28,4),activation=activation,padding='same'))
    st.session_state.model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
    st.session_state.model.add(Dropout(dropout_rate+0.05))
    st.session_state.model.add(Conv2D(128,(kernel_size,kernel_size),input_shape=(28,28,4),activation=activation,padding='same'))
    st.session_state.model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
    st.session_state.model.add(Dropout(dropout_rate+0.15))
    st.session_state.model.add(Flatten())
    st.session_state.model.add(Dense(256, activation=activation))
    st.session_state.model.add(Dropout(dropout_rate+0.3))
    st.session_state.model.add(Dense(3, activation='softmax'))
    
    # Create the CNN based on user input
    
    import visualkeras
    from PIL import ImageFont
    font = ImageFont.truetype("arial.ttf", 12)
    layered_model=visualkeras.layered_view(st.session_state.model,legend=True, font=font) 
    
    st.header('Model 3d view')
    st.image(layered_model)

    st.header('Model Summary')
    st.text(st.session_state.model.summary())
    # Display the CNN architecture
    st.header(" we can create playground like this for visualize  neural net")
    url = 'https://tensorspace.org/html/playground/lenet.html'

    html = f'<iframe src="https://tensorspace.org/html/playground/lenet.html" height="900" width="100%"></iframe>'

    st.components.v1.html(html, height=1000,width=900)

    st.image('https://i.stack.imgur.com/KVSZd.png')


if tab_name == 'Training':  
    epochs = st.slider('Epochs', 1, 10, 5) 
    batch_size = st.slider('Batch size', 16, 128, 32)
    test_size=st.slider('test split',0.1,0.4,0.2)
    import os
    from sklearn.model_selection import train_test_split

    DATA_DIR = 'dataset'
  
    # Load all images 
    all_images = []
    for img in os.listdir(DATA_DIR):
        img_array = plt.imread(os.path.join(DATA_DIR, img))
        all_images.append(img_array)

    # Extract class from each filename
    all_classes = [img.split('_')[1] for img in os.listdir(DATA_DIR)]
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt
    mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(mnist.target))
    print(all_classes)
    if st.button('Train'):
        import torch
        from torch import nn
        import torch.nn.functional as F
        # Split data into train-valid sets
        class ClassifierModule(nn.Module):
            def __init__(
                    self,
                    input_dim=mnist_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    dropout=0.5,
            ):
                super(ClassifierModule, self).__init__()
                self.dropout = nn.Dropout(dropout)

                self.hidden = nn.Linear(input_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, output_dim)

            def forward(self, X, **kwargs):
                X = F.relu(self.hidden(X))
                X = self.dropout(X)
                X = F.softmax(self.output(X), dim=-1)
                return X

        from skorch import NeuralNetClassifier
        torch.manual_seed(0)
    
        net = NeuralNetClassifier(
            ClassifierModule,
            max_epochs=epochs,
            lr=0.1,
 
        )
        st.write(net.fit(X_train, y_train))        
        # Train model on data
        
        st.header(" we can create playground like this for view training and logs")
        url = 'https://tensorspace.org/html/playground/lenet.html'

        html = f'<iframe src="https://tensorspace.org/html/playground/lenet.html" height="900" width="100%"></iframe>'

        # Display the GIF
        gif_url = "https://i.makeagif.com/media/8-24-2018/P7zznD.gif"
        st.image(gif_url, use_container_width=True)
        
if tab_name == 'Inference':

    st.write('### Draw a digit in 0-9 in the box below')
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        #background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=200,
        width=200,
        drawing_mode='freedraw',
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:

        # st.write('### Image being used as input')
        # st.image(canvas_result.image_data)
        # st.write(type(canvas_result.image_data))
        # st.write(canvas_result.image_data.shape)
        # st.write(canvas_result.image_data)
        # im = Image.fromarray(canvas_result.image_data.astype('uint8'), mode="RGBA")
        # im.save("user_input.png", "PNG")
        
        
        # Get the numpy array (4-channel RGBA 100,100,4)
        input_numpy_array = np.array(canvas_result.image_data)
        
        
        # Get the RGBA PIL image
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')

        
        # Convert it to grayscale
        input_image_gs = input_image.convert('L')
        input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200,200)
        # st.write('### Image as a grayscale Numpy array')
        # st.write(input_image_gs_np)
        
        # Create a temporary image for opencv to read it
        input_image_gs.save('temp_for_cv2.jpg')
        image = Image.open('temp_for_cv2.jpg', 0)
                    
        if os.path.exists('temp_for_cv2.jpg'):
                # Delete the file
                os.remove('temp_for_cv2.jpg')
        # Start creating a bounding box
        width, height = image.size

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        x, y, w, h = (0, 0, width, height)


        # Create new blank image and shift ROI to new coordinates
        ROI = image[y:y+h, x:x+w]
        mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
        width, height = mask.shape
    #     print(ROI.shape)
    #     print(mask.shape)
        x = width//2 - ROI.shape[0]//2 
        y = height//2 - ROI.shape[1]//2 
    #     print(x,y)
        mask[y:y+h, x:x+w] = ROI
    #     print(mask)
        # Check if centering/masking was successful
    #     plt.imshow(mask, cmap='viridis') 
        output_image = Image.fromarray(mask) # mask has values in [0-255] as expected
        # Now we need to resize, but it causes problems with default arguments as it changes the range of pixel values to be negative or positive
        # compressed_output_image = output_image.resize((22,22))
        # Therefore, we use the following:
        compressed_output_image = output_image.resize((22,22), Image.BILINEAR) # PIL.Image.NEAREST or PIL.Image.BILINEAR also performs good

        convert_tensor = torchvision.transforms.ToTensor()
        tensor_image = convert_tensor(compressed_output_image)
        # Another problem we face is that in the above ToTensor() command, we should have gotten a normalized tensor with pixel values in [0,1]
        # But somehow it doesn't happen. Therefore, we need to normalize manually
        tensor_image = tensor_image/255.
        # Padding
        tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
        # Normalization shoudl be done after padding i guess
        convert_tensor = torchvision.transforms.Normalize((0.1307), (0.3081)) # Mean and std of MNIST
        tensor_image = convert_tensor(tensor_image)
        # st.write(tensor_image.shape) 
        # Shape of tensor image is (1,28,28)
        


        # st.write('### Processing steps:')
        # st.write('1. Find the bounding box of the digit blob and use that.')
        # st.write('2. Convert it to size 22x22.')
        # st.write('3. Pad the image with 3 pixels on all the sides to get a 28x28 image.')
        # st.write('4. Normalize the image to have pixel values between 0 and 1.')
        # st.write('5. Standardize the image using the mean and standard deviation of the MNIST_plus dataset.')

        # The following gives noisy image because the values are from -1 to 1, which is not a proper image format
        im = Image.fromarray(tensor_image.detach().cpu().numpy().reshape(28,28), mode='L')
        im.save("processed_tensor.png", "PNG")
        # So we use matplotlib to save it instead
        plt.imsave('processed_tensor.png',tensor_image.detach().cpu().numpy().reshape(28,28), cmap='gray')

        # st.write('### Processed image')
        # st.image('processed_tensor.png')
        # st.write(tensor_image.detach().cpu().numpy().reshape(28,28))


        device='cpu'
        ### Compute the predictions
        with torch.no_grad():
            # input image for network should be (1,1,28,28)
        
            output0 = Network(torch.unsqueeze(tensor_image, dim=0).to(device=device))
            # Need to apply Softmax here to get probabilities
            m = torch.nn.Softmax(dim=1)
            output0 = m(output0)
            # st.write(output0)
            certainty, output = torch.max(output0[0], 0)
            certainty = certainty.clone().cpu().item()
            output = output.clone().cpu().item()
            certainty1, output1 = torch.topk(output0[0],3)
            certainty1 = certainty1.clone().cpu()#.item()
            output1 = output1.clone().cpu()#.item()
    #     print(certainty)
        st.write('### Prediction') 
        st.write('### '+str(output))

        st.write('## Breakdown of the prediction process:') 

        st.write('### Image being used as input')  
        st.image(canvas_result.image_data)

        st.write('### Image as a grayscale Numpy array')
        st.write(input_image_gs_np)

        st.write('### Processing steps:')
        st.write('1. Find the bounding box of the digit blob and use that.')
        st.write('2. Convert it to size 22x22.')
        st.write('3. Pad the image with 3 pixels on all the sides to get a 28x28 image.')
        st.write('4. Normalize the image to have pixel values between 0 and 1.')
        st.write('5. Standardize the image using the mean and standard deviation of the MNIST training dataset.')

        st.write('### Processed image')
        st.image('processed_tensor.png')

        st.write('### Prediction') 
        st.write(str(output))
        st.write('### Certainty')    
        st.write(str(certainty1[0].item()*100) +'%')
        st.write('### Top 3 candidates')
        st.write(str(output1))
        st.write('### Certainties')    
        st.write(str(certainty1*100))


        st.header(" we can create playground like this for inference")
        url = 'https://tensorspace.org/html/playground/lenet.html'

        html = f'<iframe src="https://tensorspace.org/html/playground/lenet.html" height="900" width="100%"></iframe>'
