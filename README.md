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

