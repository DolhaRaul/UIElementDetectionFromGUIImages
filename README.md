# UI Element Detection from GUI Images

Graphical User Interfaces (GUIs) are among the most common types of interfaces, enabling users to
navigate various websites quickly and efficiently. This project focuses on the precise identification
of up to 16 different elements within GUI images, a crucial capability for applications such as automated testing.

# Table of Contents
1. [Data Analysis](#data-analysis)
2. [Development and Assessment of the Algorithm](#development-and-assessment-of-the-algorithm)
3. [Conclusions](#conclusions)

## Data Analysis

The input data that the machine learning algorithm will take and process
represents a set of 121 images, more precisely 121 screenshots of various 
pages of a website, having the size of  1920 × 940 pixels or 1902 × 922
pixels, based on the article [DSCO24](https://www.scitepress.org/Documents/2024/126326/). Thus, within the project carried out using
the Python programming language there is a folder called "data", which in turn will
account, has a folder called "images" in which the images are stored in the format "ImageName".png that will be processed by the algorithm, see figure below. The 121 images are segmented so that each element of the page is included in the screenshot
is framed in a rectangle, whose representation is given to the file from the project structure called "annotations.csv" uniquely by the coordinates of their right-bottom and top-left points of the respective rectangle. 
They are in the table below
show the 16 types of elements (classes) and the number of occurrences of each, all this totaling 15123 elements for the algorithm to identify. 

| Element's type (class) | Number |
|---------------------------|-------|
| Icon                      | 5704  |
| TextLabel                 | 4298  |
| MenuItem                  | 1823  |
| Row                       | 813   |
| Button                    | 757   |
| SubmenuItem               | 685   |
| NavigationItem            | 424   |
| DropdownItem              | 379   |
| InputField                | 149   |
| SectionTitle              | 145   |
| TitleBar                  | 121   |
| Menu                      | 2154  |
| WorkingArea               | 121   |
| VerticalMenu              | 110   |
| NavigationMenu            | 100   |
| TableHeader               | 58    |

Below is information on where the dataset images for our model are stored and how the annotations for the classes within each image are organized.
<p align="center">
  <img src="https://github.com/user-attachments/assets/15bb887d-26e1-4fe7-af66-7bb854f3b320" alt="Imagine1_UIElemDetection" width="45%">
  <img src="https://github.com/user-attachments/assets/e35a1171-1b1c-493d-b8fb-5280faef0720" alt="Imagine2_UIElemDetection" width="45%">
</p>

Next, to enable the algorithm to be trained, tested, and evaluated, we need to extract this data from the previously mentioned file. This process involves determining the coordinates of the center of the bounding box, which are the rectangular shapes that frame the elements on the page. To be able to be processed
by the YOLO V8 algorithm, the **normalization process** was applied, an operation that was done based on the scientific article [ZAST21](https://dsa21.techconf.org/download/DSA2021_FULL/pdfs/DSA2021-1sP33wTCujRJmnnXDjv3mG/439100a508/439100a508.pdf)

The algorithm retrieves the data from the "labels" folder, which contains a text file, sub
the "image name" format. txt that contains, on the first column, the corresponding number of each
element type (of each class) from table ˘ 2.1 as follows: icon − > 0, TextLabel − > 1,
MenuItem − > 2, ... , NavigationMenu − > 14, TableHeader − > 15, on the rest of the other columns being present which represent numbers from the range ˘ [0, 1] which represent
data normalized by the process described above, as can be observed in the figure previously attached.

Part of the pre-trained algorithms used to compare the results uses the COCO dataset for training, which contains 330,000 images, among
which 200,000 have annotations that allow object detection, segmentation and framing,
including 80 categories of objects. ˆ
In our problem, the pre-trained algorithms
used, they used 90% of the images for training, and they were used for evaluation
the remaining 10% of the images.

## Development and Assessment of the Algorithm

To solve the given problem, we chose to use the YOLO algorithm, more precisely the V8 version of it, a relatively new and improved version compared to its predecessors, which offers fast very good results.  YOLOv8 is the second most
new model from the YOLO (You Only Look Once) series of algorithms - the best-known family of object detection and classification models in the Computer field
Vision (CV), discovered in 2015 by the researcher Joseph Redmon and his collaborators. 
The YOLOv8 architecture (see figure below) represents an evolution of the previous YOLO models, using a convolutional neural network divided into two main parts: the trunk" and the "head". The trunk is based on a modified version of the architecture
of the CSPDarknet53 (neural network for image detection) structure, consisting of
53 convolutional layers. The head comprises several convolutional layers followed by fully connected layers responsible for predicting the bounding boxes
of delimitation and the probabilities of the appearance of the classes. YOLOv8 integrates a mechanism in the head of the network and a pyramidal network for object detection at multi-scale
ple, allowing it to focus on various parts of an image and detect
objects of different sizes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e60f0f9e-02df-445b-b046-56bc7c404a4b" alt="Imagine3_UIElemDetection" width="45%">
  <img src="https://github.com/user-attachments/assets/d3a1c612-fb46-447f-93a0-16376a4c2b71" alt="Imagine4_UIElemDetection" width="45%">
</p>



YOLOv8 presents several pretrained models( see [Yolo Models List Information](https://medium.com/@beyzaakyildiz/what-is-yolov8-how-to-use-it-b3807d13c5ce)), from which we make use of the following: **nano**, **small**, **medium**

In the current problem, the program is parameterized, offering the user an interface through which to choose which version from the three that are
offered (nano, small, medium) to use it for testing and evaluation. In order to
could compare the results obtained by running some pre-trained algorithms with others
untrained, we will define the metrics we use in making the comparisons:

1. **IoU metric**: It is given as the ratio between the number of pixels in the intersection of the real bounding box with the bounding box returned by the algorithm used and the number of pixels in the meeting of the two bounding boxes (see figure above ).

2. **Metric precision (accuracy)**:  
   `precision = no. TP cases / (no. TP + FP cases)`

3. **Metric recall (sensitivity)**:  
   `recall = no. TP cases / (no. TP + FN cases)`

4. **Accuracy metric**: Defined as the ratio of the number of correctly classified data to the total number of data:  
   `accuracy = (no. cases TP + TN) / (no. cases TP + TN + FP + FN)`

5. **F1 metric**:  
   `F1 = 2 * recall * precision / (recall + precision)'

6. **Average Accuracy (AP)**:  
   `AP = (recall + precision) / 2`

7. **mean Average Precision (mAP)**: Given the total number of classes and `AP_i` as the average precision of class `i`, then:  
   `mAP = (Σ AP_i) / n` (where the sum is done for `i` from 1 to `n`)

Another feature that offers us important information regarding how well the algorithm identifies the elements is the **normalized confusion matrix**

The program also allows you to choose the **model's version** of YOLO V8 to enter
on the data set, and choosing the **batch** (the number of images to be processed
concurrently) set by default to 16, the number of **epochs** ( represents the number of
flows of the data set by the algorithm used) set by default to 100, but we 
I trained, validated and evaluated them most of the time on smaller numbers (usually
30 epochs), because the running time for 100 epochs can exceed 5 hours.
In addition to these parameters, we can also choose the size of the **output image** (which
must be of the form  n × n, n multiple of 32)

<hr>
Next, we will present the results obtained using pre-trained algorithms
using the data set available:

**1**. If we use the nano algorithm, for 30 epochs, we obtain the matrix
of confusion from figure below, from which we can see that some elements
are found correctly in 100% proportion (menu, vertical menu), while
some are not identified at all (section title, title bar), the metrics values being good overall.


<p align="center">
  <img src="https://github.com/user-attachments/assets/a05e5377-1b82-4b06-87bf-e68614f09ea2" alt="Image1" width="40%"> 
  <img src="https://github.com/user-attachments/assets/60e48920-4515-4e86-b5ab-380ea80d015c" alt="Image2" width="40%">  
</p>

To test the model and make predictions, the program includes a dedicated menu where users can select the **model** for prediction (in this case, the previously trained nano model with 30 epochs), an **image** to process (one not included in the training dataset), and the **specific classes to identify** in the image (with all classes selected by default). For a more challenging test, we chose the class with the highest number of occurrences, the Icon class (76 instances), resulting in 45 elements being identified. The output image is shown below.
![image](https://github.com/user-attachments/assets/f0167b66-8ddd-4159-9f4a-0c1a45f83337)
<div align="center">
  <em>Predictions</em> for <strong>Icon class</strong> for image found in <em>/data/images/Imagine1</em>
</div>

**2**. Next, we will present the results obtained by the small version (which requires a slightly longer running time - **approximately 2 hours**), but which provides more accurate results over the **same number of epochs**. The obtained confusion matrix
is shown in figure below, which we can see are significantly better
than in the nano30 version, having **4 classes** found correctly in a proportion of 100%
(menu, vertical menu, navigation menu, working area), **most classes**
having a score of over **0.9**, none of them being unidentified.



