#pip install pydot
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
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
    tabs = st.radio("Select  Tab:", ["Home page", "Prepare datset", "Create Neural net","Training",'Inference'], key='tabs', index=0)

# Update the selected tab
st.session_state.selected_tab = tabs

# Get the active tab name
tab_name = st.session_state.selected_tab

def new_line():
    st.markdown("<br>", unsafe_allow_html=True)

if tab_name == 'Home page':

    # Title Page
    st.markdown("<h1 style='text-align: center; '>📚 Handwritten Digit Recognition</h1>", unsafe_allow_html=True)

    new_line()
    # Display the GIF
    gif_url = "https://i.makeagif.com/media/8-24-2018/P7zznD.gif"
    st.image(gif_url)
    new_line()

    new_line()
    st.markdown("Welcome to Handwritten Digit Recognition! This tab is designed to help you understand the key concepts of Data Preparation and CNN architecture. Please select a topic below to get started.", unsafe_allow_html=True)
    new_line()

    # Tabs
    tab_titles = ['🗺️ Overview 󠀠 󠀠 󠀠', '🧭 EDA 󠀠 󠀠 󠀠', "‍📀‍‍‍‍ Checking blur and duplicates 󠀠󠀠 󠀠 󠀠", "🧬 Scaling & Transformation 󠀠 󠀠 󠀠", "✂️ Splitting the Data 󠀠 󠀠 󠀠", "🧠 CNN architecture 󠀠 󠀠 󠀠"]
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:
        new_line()

        st.markdown("<h2 style='text-align: center; '>🗺️ Overview</h2>", unsafe_allow_html=True)
        new_line()
        
        st.markdown("""
        When you are building a Machine Learning model, you need to follow a series of steps to prepare the data and build the model. The following are the key steps in the Machine Learning process:
        
        - **📦 Data Collection**: is the process of collecting the data from various sources such as CSV files, databases, APIs, etc. One of the most famous websites for datasets is [**Kaggle**](https://www.kaggle.com/). <br> <br>
        - **🧹 Data Cleaning**: is the process of cleaning the data by removing duplicates, handling blur images etc. This step is very important because at most times the data is not clean. <br> <br>
        - **⚙️ Data Preprocessing**: is the process of transforming the data into a format that is suitable for analysis. This includes handling blur images, converting images to array , scaling and transformation, etc. <br> <br>
        - **💡 Feature Engineering**: is the process that manipulate with the features itselfs. It consists of multiple steps such as feature extraction, feature transformation, and feature selection. <br> <br>
        - **✂️ Splitting the Data**: is the process of splitting the data into training, validation, and testing sets. The training set is used to train the model, the validation set is used to tune the hyperparameters, and the testing set is used to evaluate the model. <br> <br>
        - **🧠 Building CNN Models**: is the process of building the CNN models. There are many CNN architecture  that can be used for Hand written bclassification  tasks. Some of the most famous models are Linear Regression, Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Neural Networks. <br> <br>
        - **⚖️ Evaluating CNN Models**: is the process of evaluating the CNN models using various metrics such as accuracy, precision, recall, F1 score, and many more for classification tasks . <br> <br>
        - **📐 Tuning Hyperparameters**: is the process of tuning the hyperparameters(inputs) of the CNN models to get the best model. There are many hyperparameters that can be tuned for each model such as  the number of layers and neurons for Neural Networks, and many more. <br> <br>
        """, unsafe_allow_html=True)
        new_line()
        
    with tabs[1]:
        new_line()
        st.markdown("<h2 style='text-align: center; ' id='eda'>🧭 Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
        # half_divider()
        new_line()
        st.markdown("Exploratory Data Analysis (EDA) is the process of analyzing data sets to summarize their main characteristics, often with visual methods. EDA is used for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. It is an important step in the Data Preparation process. EDA is also the first step in the Machine Learning process. It is important to understand the data before building a model. This will help you to choose the right model and avoid errors. EDA is also used to find patterns, spot anomalies, test hypothesis and check assumptions with the help of summary statistics and graphical representations.", unsafe_allow_html=True)
        new_line()


        st.markdown("<h6 > The following are some of the key steps in EDA:", unsafe_allow_html=True)
        st.markdown("- **Data Collection:** This is the first step in EDA. Data can be collected from various sources such as CSV files, databases, APIs, etc.", unsafe_allow_html=True)
        st.markdown("- **Data Cleaning:** This is the process of cleaning the data by removing duplicates, handling missing values, handling outliers, etc.", unsafe_allow_html=True)
        st.markdown("- **Data Preprocessing:** This is the process of transforming the data into a format that is suitable for analysis. This includes handling categorical features, handling numerical features, scaling and transformation, etc.", unsafe_allow_html=True)
        st.markdown("- **Data Visualization:** This is the process of visualizing the data using various plots such as bar plots, histograms, scatter plots, etc.", unsafe_allow_html=True)
        st.markdown("- **Data Analysis:** This is the process of analyzing the data using various statistical methods such as mean, median, mode, standard deviation, etc.", unsafe_allow_html=True)
        st.markdown("- **Data Interpretation:** This is the process of interpreting the data to draw conclusions and make decisions.", unsafe_allow_html=True)
        new_line()

    with tabs[2]:

        new_line()
        st.markdown("<h2 align='center'> ‍📀‍‍‍‍ Checking blur and duplicates </h1>", unsafe_allow_html=True)
        
        # What is Missing Values?
        new_line()
        st.markdown("blur and duplicates images are common in real-world datasets. Blur and duplicate images can be caused by many reasons, such as human errors, data collection errors, or data processing errors. Due to this images it can cause problems in the CNN process leads to degredation of performance of cnn. So, we need to remove blur and duplicate images before we can use the data in the CNN process.", unsafe_allow_html=True)
        new_line()

        # Why we should handle the missing values?
        st.markdown("#### ❓ Why we should remove blur and duplicate images?")
        st.markdown("Due to this images it can cause problems in the CNN process leads to degredation of performance of cnn. So, we need to remove blur and duplicate images before we can use the data in the CNN process.", unsafe_allow_html=True)
        new_line()

        st.markdown("#### 🌎 In General")
        st.markdown("**Drop the duplicate images:** We can drop the duplicate images from the data. That is the easiest way to handle duplicate images")
        st.markdown("- **Remove the blurness in images**: There will be many memthods avialable to remove blur in images.we can use those method for removing blurness of images")
        new_line()
    
    with tabs[3]:

        new_line()
        st.markdown("<h2 align='center'> 🧬 Scaling & Transformation </h1>", unsafe_allow_html=True)

        # What is Scaling & Transformation?
        new_line()
        st.markdown(" :green[Data Scaling] is a method for scaling the data to a specific range, that is becuase the data can have different ranges and when a feature has a higher range, then it will have a higher impact on the model and it will add **bias**. So, we need to scale the data to a specific range.")
        st.markdown(" :green[Data Transformation] is a method for transforming the data to a specific distribution, that is becuase the data can have different distributions and when a feature has a different distribution, then it will have a higher impact on the model and it will add **bias**. So, we need to transform the data to a specific distribution. This method applied especially when the data has outliers and have high skewness.")
        new_line()

        # Why we should scale the data?
        st.markdown("##### 📏 Why we should scale the data?")
        st.markdown("Scaling the data is important for some CNN algorithms. That is because some layers cnn  algorithms are sensitive to the range of the data. So we need to scale the data before we can use the data in CNN architecture.", unsafe_allow_html=True)
        new_line()

        # Why we should transform the data?
        st.markdown("##### ➰ Why we should transform the data?")
        st.markdown("Transforming the data is important for some CNN algorithms. That is because some layers od CNN algorithms are sensitive to the distribution of the data. So, we need to transform the data before we can use the data", unsafe_allow_html=True)
        # new_line()
        st.divider()
    
    with tabs[4]:
        
        new_line()
        st.markdown("<h2 align='center'> ✂️ Splitting The Data </h1>", unsafe_allow_html=True)
        new_line()

        # What is Splitting The Data?
        st.markdown("Splitting The Data is the process to split the data into three parts: **Training Data**, **Valication Data**, and **Testing Data**. Splitting the data is very important step in the Machine Learning process, this is beause you want to evaluate the model on an unseen data. So we need to split the data we have into 3 part:")
        st.markdown("1. :green[Training Data:] This data is used to train the model. It must have the highest number of rows. The percentage of the training data is 60% to 80% of the total number of rows.")
        st.markdown("2. :green[Validation Data:] This data is used for Hyperparameter Tuning. It must have the lowest number of rows. The percentage of the validation data is 10% to 20% of the total number of rows.")
        st.markdown("3. :green[Testing Data:] This data is used to final evaluation for the model. It must have the second highest number of rows. The percentage of the testing data is 20% to 30% of the total number of rows.")

    with tabs[5]:
        new_line()
        st.markdown("<h2 align='center'> 🧠 CNN architecture</h1>", unsafe_allow_html=True)
        new_line()
        
        new_line()
        st.title('Convolutional Neural Networks (CNN)')
        new_line()

        st.markdown('''Imagine being in a zoo trying to recognize if a given animal is a cheetah or a leopard. As a human, your brain can effortlessly analyze body and facial features to come to a valid conclusion. In the same way, Convolutional Neural Networks (CNNs) can be trained to perform the same recognition task, no matter how complex the patterns are. This makes them powerful in the field of computer vision.This conceptual CNN tutorial will start by providing an overview of what CNNs are and their importance in machine learning. Then it will walk you through a step-by-step implementation of CNN in TensorFlow Framework 2.''')
        
        new_line()
        st.title('What is a CNN?')
        new_line()

        st.markdown('''A Convolutional Neural Network (CNN or ConvNet) is a deep learning algorithm specifically designed for any task where object recognition is crucial such as image classification, detection, and segmentation.  Many real-life applications, such as self-driving cars, surveillance cameras, and more, use CNNs.''')
        
        new_line()
        st.title('The importance of CNNs')
        new_line()
        
        st.markdown('These are several reasons why CNNs are important, as highlighted below:')
        st.markdown('''  *Unlike traditional machine learning models like SVM and decision trees that require manual feature extractions, CNNs can perform automatic feature extraction at scale, making them efficient. ''')
        st.markdown('''  *The convolutions layers make CNNs translation invariant, meaning they can recognize patterns from data and extract features regardless of their position, whether the image is rotated, scaled, or shifted.''')
        st.markdown('''  *Multiple pre-trained CNN models such as VGG-16, ResNet50, Inceptionv3, and EfficientNet are proved to have reached state-of-the-art results and can be fine-tuned on news tasks using a relatively small amount of data.''')
        st.markdown('''  *CNNs can also be used for non-image classification problems and are not limited to natural language processing, time series analysis, and speech recognition.''')
        
        new_line()
        st.title('Architecture of a CNN')
        new_line()
        
        st.markdown('''CNNs architecture tries to mimic the structure of neurons in the human visual system composed of multiple layers, where each one is responsible for detecting a specific feature in the data.  As illustrated in the image below, the typical CNN is made of a combination of four main layers:''')
        st.markdown('''   *Convolutional layers''')
        st.markdown('''   *Rectified Linear Unit (ReLU for short)''')
        st.markdown('''   *Pooling layers''')
        st.markdown('''   *Fully connected layers''')

        st.markdown('''Let's understand how each of these layers works using the following example of classification of the handwritten digit.''')
        
        st.image('https://images.datacamp.com/image/upload/v1681492916/Architecture_of_the_CN_Ns_applied_to_digit_recognition_0d403dcf68.png')

        new_line()
        st.title('Convolution layers')
        new_line()

        st.markdown('''This is the first building block of a CNN. As the name suggests, the main mathematical task performed is called convolution, which is the application of a sliding window function to a matrix of pixels representing an image. The sliding function applied to the matrix is called kernel or filter, and both can be used interchangeably.''')
        st.markdown('''In the convolution layer, several filters of equal size are applied, and each filter is used to recognize a specific pattern from the image, such as the curving of the digits, the edges, the whole shape of the digits, and more. ''')
        st.markdown('''Let's consider this 32x32 grayscale image of a handwritten digit. The values in the matrix are given for illustration purposes.''')
        
        st.image('https://images.datacamp.com/image/upload/v1681493488/Illustration_of_the_input_image_and_its_pixel_representation_9185e3f876.png')
        
        st.markdown('''Also, let's consider the kernel used for the convolution. It is a matrix with a dimension of 3x3. The weights of each element of the kernel is represented in the grid. Zero weights are represented in the black grids and ones in the white grid.''')
        
        st.markdown('''green[Do we have to manually find these weights?]''')

        st.markdown('In real life, the weights of the kernels are determined during the training process of the neural network.')
        st.markdown('Using these two matrices, we can perform the convolution operation by taking applying the dot product, and work as follows: ')

        st.markdown('   1.Apply the kernel matrix from the top-left corner to the right.')
        st.markdown('   2.Perform element-wise multiplication. ')
        st.markdown('   3.Sum the values of the products.')
        st.markdown('   4.The resulting value corresponds to the first value (top-left corner) in the convoluted matrix')
        st.markdown('   5.Move the kernel down with respect to the size of the sliding window.')
        st.markdown('   6.Repeat from step 1 to 5 until the image matrix is fully covered.')

        st.markdown('The dimension of the convoluted matrix depends on the size of the sliding window. The higher the sliding window, the smaller the dimension.')

        st.image('https://images.datacamp.com/image/upload/v1681493568/image13_Application_of_the_convolution_task_using_a_stride_of_1_with_3x3_kernelpng_cc7119e1ff.png')
        
        st.markdown('Another name associated with the kernel in the literature is feature detector because the weights can be fine-tuned to detect specific features in the input image. ')
        st.markdown('For instance: ')
        st.markdown('   * Averaging neighboring pixels kernel can be used to blur the input image.')
        st.markdown('   * Subtracting neighboring kernel is used to perform edge detection.')

        st.markdown('The more convolution layers the network has, the better the layer is at detecting more abstract features.')
        
        new_line()
        st.title('Activation function')
        new_line()

        st.markdown('A ReLU activation function is applied after each convolution operation. This function helps the network learn non-linear relationships between the features in the image, hence making the network more robust for identifying different patterns. It also helps to mitigate the vanishing gradient problems.')
        st.image('https://149695847.v2.pressablecdn.com/wp-content/uploads/2019/01/ann-act.gif')
        
        new_line()
        st.title('Pooling layer')
        new_line()

        st.markdown('The goal of the pooling layer is to pull the most significant features from the convoluted matrix. This is done by  applying some aggregation operations, which reduces the dimension of the feature map (convoluted matrix), hence reducing the memory used while training the network.  Pooling is also relevant for mitigating overfitting.')
        
        st.markdown('The most common aggregation functions that can be applied are: ')
        st.markdown('   * Max pooling which is the maximum value of the feature map')
        st.markdown('   * Sum pooling corresponds to the sum of all the values of the feature map')
        st.markdown('   * Average pooling is the average of all the values.')

        st.markdown('Below is an illustration of each of the previous example: ')

        st.image('https://images.datacamp.com/image/upload/v1681493690/Application_of_max_pooling_with_a_stride_of_2_using_2x2_filter_eb516c36dc.png')

        st.markdown('Also, the dimension of the feature map becomes smaller as the polling function is applied. ')
        st.markdown('The last pooling layer flattens its feature map so that it can be processed by the fully connected layer.')

        new_line()
        st.title('Fully connected layers')
        new_line()

        st.markdown('These layers are in the last layer of the convolutional neural network,  and  their inputs correspond to the flattened one-dimensional matrix generated by the last pooling layer. ReLU activations functions are applied to them for non-linearity. ')
        st.markdown('Finally, a softmax prediction layer is used to generate probability values for each of the possible output labels, and the final label predicted is the one with the highest probability score.')
        
        st.image('https://open-instruction.com/loading/2021/05/e.png')


Network= torch.load(r'D:\assignment\vizura\model_torch_MNIST_CNN_99_1_streamlit.chk')

if 'uploading_way' not in st.session_state:
        st.session_state['uploading_way'] = None


if tab_name == 'Prepare datset':

        # Dataframe selection
    st.markdown("<h2 align='center'> <b> Getting Started", unsafe_allow_html=True)
    new_line()
    st.write("The first step is to create  data. You can upload your data in two ways: **create using canvas**, **Select from Ours**,  In all ways the data should be a image file.")
    new_line()

    uploading_way = st.session_state.uploading_way
    col1, col2 = st.columns(2,gap='large')
    def upload_click(): st.session_state.uploading_way = "create"
    col1.markdown("<h5 align='center'> create dataset", unsafe_allow_html=True)
    col1.button("create dataset", key="create dataset", use_container_width=True, on_click=upload_click)

    # Select    
    def select_click(): st.session_state.uploading_way = "select"
    col2.markdown("<h5 align='center'> Select from Ours", unsafe_allow_html=True)
    col2.button("Select from Ours", key="select_from_ours", use_container_width=True, on_click=select_click)

    if uploading_way == "create":
   
            # Specify canvas parameters in application
            stroke_width1 = st.sidebar.slider("Stroke width1: ", 1, 25, 9)
            # Create a canvas component
            st.write("Create a dataset for handwritten classification. draw atleast 10 images per class")
            class_counts = {}  
            st.markdown('Enter number of output class (class means number of letters cnn needs to predict)')
            # Number of classes
            num_classes = st.number_input("Number of Classes", min_value=1, value=3)
            st.markdown('Provide name for output class')

            st.markdown('   Ex: lets assume you have selected 3 class then yopu have provide names in commas seperated= one,two,three   then press enter')

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
            st.markdown('Select class name from dropdown')

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
                        image = cv2.imread('temp_for_cv2.jpg', 0)
                            
                        if os.path.exists('temp_for_cv2.jpg'):
                                # Delete the file
                                os.remove('temp_for_cv2.jpg')
                        # Start creating a bounding box
                        height, width = image.shape
                        x,y,w,h = cv2.boundingRect(image)
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


    # Select
    elif uploading_way == "select":
                selected = st.selectbox("Select Dataset", ["Select", "Mnist Dataset",])

    



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
    
    st.markdown('click on upload symbol in below space and upload image let the magic to happen')
    st.image('input image.png')

    url = 'https://poloclub.github.io/cnn-explainer/'

    html = f'<iframe src="https://poloclub.github.io/cnn-explainer/" height="900" width="100%"></iframe>'

    st.components.v1.html(html, height=1000,width=900)

    st.title("CNN Builder")
    # User inputs


   
    dropout_rate = st.slider("Dropout rate:", 0.0, 0.5, 0.5)
    activation = st.selectbox('Activation', ['relu', 'tanh'])
    kernel_size = st.slider('Kernel size', 2, 8, 2) 
    pool_size = st.slider('Pool size', 2, 8, 2)
    num_output_classes = st.number_input("Number of output classes:", min_value=1, value=10)

   
    
    model = Sequential()
    model.add(Conv2D(64,(4,4),input_shape=(28,28,4),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(4,4),input_shape=(28,28,4),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(128,(2,2),input_shape=(28,28,4),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128,(4,4),input_shape=(28,28,4),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(3, activation='softmax'))
    
    # Create the CNN based on user input
    
    import visualkeras
    from PIL import ImageFont
    font = ImageFont.truetype("arial.ttf", 12)
    layered_model=visualkeras.layered_view(model,legend=True, font=font) 
    
    st.header('Model 3d view')
    st.image(layered_model)


    # Display the CNN architecture
    st.header(" we can create playground like this for visualize  neural net")


    st.image('https://i.stack.imgur.com/KVSZd.png')


if tab_name == 'Training':  
    epochs = st.slider('Epochs', 1, 10, 5) 
    batch_size = st.slider('Batch size', 16, 128, 32)
    test_size=st.slider('test split',0.1,0.4,0.2)
    import os
    from sklearn.model_selection import train_test_split

    DATA_DIR = 'dataset'
  
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np 

    # Load images
    X = np.load('mnist_images.npy')

    # Load labels 
    y = np.load('mnist_labels.npy', allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(y)) 

    if st.button('Train'):
                
        # Train model on data
        
        st.header(" we can create playground like this for view training and logs")
        url = 'https://tensorspace.org/html/playground/trainingLeNet.html'

        html = f'<iframe src="https://tensorspace.org/html/playground/trainingLeNet.html" height="900" width="100%"></iframe>'
        st.components.v1.html(html, height=1000,width=900)


        
if tab_name == 'Inference':
    st.title("💡 Inference")

    st.write('### Draw a digit in 0-9 in the box below')

    url = 'https://tensorspace.org/html/playground/lenet.html'

    html = f'<iframe src="https://tensorspace.org/html/playground/lenet.html" height="900" width="100%"></iframe>'
    
    st.components.v1.html(html, height=1000,width=900)
