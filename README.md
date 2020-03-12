### ***This project is a basic American Sign Language interpreter which applies in a few letters(A,B,C,D,F,G,H,L) implemented with OpenCV and Keras.***

<br>

The DataSet was created by me and it consists of 4800 images.
  
  - 600 for each letter tottaly.
    - 500 for training
    - 50 for testing
    - 50 for validating
    
Every image has been going through blurring (Median and Gaussian) and thresholing with [Otsu's Method](https://en.wikipedia.org/wiki/Otsu%27s_method).
  
<br>

A image sample from the dataset and specifically for the letter L is:
<br>
![file_37](https://user-images.githubusercontent.com/37080724/76479807-a5626000-6415-11ea-8dff-cc7d688fdfa6.jpg)

<br>
The sequential CNN that was trained is shown below:
<br>

![SeqModel](https://user-images.githubusercontent.com/37080724/76480720-3afeef00-6418-11ea-8321-9bfc1102db0e.png)

<br>

*A fully functional demo will be uploaded soon!*

**Next Steps:**
- [ ] Writing letters/words in the live feed using OpenCV.
- [ ] Convert the above project for the Greek Sign Language.
