# UI Element Detection from GUI Images

Graphical User Interfaces (GUIs) are among the most common types of interfaces, enabling users to
navigate various websites quickly and efficiently. This project focuses on the precise identification
of up to 16 different elements within GUI images, a crucial capability for applications such as automated testing.

# Table of Contents
1. [Data Analysis](#data-analysis)
2. [Development and assesment of the algorithm](#study)
3. [Conclusions](#conclusion)

 ## Data Analysis

The input data that the machine learning algorithm will take and process
represents a set of 121 images, more precisely 121 screenshots of various 
pages of a website, having the size of  1920 × 940 pixels or 1902 × 922
pixels, based on the article [DSCO24](https://www.scitepress.org/Documents/2024/126326/). Thus, within the project carried out using
the Python programming language there is a folder called "data", which in turn will
account, has a folder called "images" in which the images are stored in the format "nu meImage".png that will be processed by the algorithm, see figure 2.1. The 121 images are segmented so that each element of the page is included in the screenshot
is framed in a rectangle, whose representation is given to the file from the project structure called "annotations.csv" uniquely by the coordinates of their right-bottom and top-left points of the respective rectangle. ˆ
They are in the table below
show the 16 types of elements (classes) and the number of occurrences of each
