# HW to Chapter 18 “Neural Style Transfer”

## Non-programming Assignment

### Q1. What is face verification and how does it work?

### Answer:

Face verification is a biometric process that compares a presented face to a single face image to verify the identity of an individual. It is a "one-to-one" matching process used to answer the question: "Is this the person they claim to be?" Here's a closer look at how it typically works:   

1. Face Detection   
The first step involves detecting a face within the provided image or video frame. This process identifies the presence and location of a face in any given input.    

2. Feature Extraction     
Once a face is detected, the system extracts unique facial features from the image. These features include various points and contours of the face, such as the eyes, nose, mouth, and jawline, as well as texture patterns. Advanced face verification systems utilize deep learning models, particularly convolutional neural networks (CNNs), to generate a facial signature—a numerical representation of the face's features, often referred to as an embedding. These embeddings capture deep, abstract features of the face that are distinctive to an individual.   

3. Comparison   
The extracted facial features or embeddings from the presented face are then compared to those of a known face image (the one stored in the database or provided for verification). This comparison is usually quantified through similarity measures such as Euclidean distance, cosine similarity, or other distance metrics. The idea is to compute how "close" the presented face is to the stored face in terms of their feature representations.   

4. Decision Making  
Based on the similarity score obtained from the comparison, a decision is made. If the score indicates that the presented face and the stored face are sufficiently similar (beyond a predefined threshold), the system verifies the identity of the individual. The threshold is set to balance between false acceptances (verifying the wrong person) and false rejections (failing to verify the correct person) according to the application's security requirements.   

#### Implementation Considerations:

Threshold Settings:    
The threshold for deciding whether a match is successful can be adjusted according to the security requirements of the system. A lower threshold may reduce false rejections but increase false acceptances, and vice versa.   
Security and Privacy:    
Face verification systems must handle biometric data securely, ensuring that facial data is stored and processed in a way that protects individuals' privacy.    
Performance Factors:    
The accuracy of face verification can be affected by various factors, including lighting conditions, facial expressions, aging, and occlusions (e.g., glasses or masks). Modern systems use robust models trained on diverse datasets to improve performance under a wide range of conditions.    

Face verification is widely used in various applications, including security systems, access control, banking, and mobile device unlocking, offering a convenient and secure method to verify individuals' identities based on their unique facial features.

### Q2. Describe the difference between face verification and face recognition?

### Answer:   

Face verification and face recognition are related but distinct processes within the broader field of biometric identification, each serving different purposes and employing different methodologies.    

1. Face Verification (1:1 Matching)   

Purpose:    
Face verification, also known as identity authentication, involves confirming or denying a person's claimed identity. It answers the question: "Is this person who they claim to be?"    
Process:     
In face verification, a single facial image is compared to a specific face image or template stored in a database to verify the claimed identity. The system performs a one-to-one (1:1) match to ascertain whether the two images belong to the same person.     
Applications:     
This method is commonly used in security systems for access control (e.g., unlocking smartphones, secure entry to buildings, or logging into secure systems), where users must prove their identity against a pre-registered profile.

2. Face Recognition (1:N Matching)     
Purpose:      
Face recognition, also known as identity identification, is about identifying an individual from a pool of many faces. It seeks to answer the question: "Who is this person?" without prior claims about the individual's identity.     
Process:     
In face recognition, a facial image is compared against a database containing multiple faces to find a match. This process involves one-to-many (1:N) matching, where the system scans the database to find if there already exists a record that matches the given face.
Applications:      
Face recognition is used in scenarios requiring the identification of individuals among many, such as surveillance systems, finding missing persons, tagging friends in social media photos, and managing customer identities in marketing analytics.

#### Key Differences    

Matching Approach:     
Verification is a one-to-one matching process checking if the presented face matches a specific known face. Recognition is a one-to-many matching process, determining if the presented face matches any face within a larger dataset.
Objective:      
Verification authenticates a claimed identity, essentially a binary yes/no decision. Recognition identifies an unknown face from a set of known faces, potentially returning the identity of the face.
Use Cases:      
Verification is used for personal authentication tasks, while recognition is applied in scenarios requiring the identification or cataloging of individuals from a group or database.    

In summary, the main distinction between face verification and face recognition lies in their objectives and the nature of their matching processes—verification confirms a claimed identity by comparing two faces, while recognition identifies a face from a group of many faces.

### Q3. How do you measure similarity of images?

### Answer:


### Q4. Describe Siamese networks.

### Answer:


### Q5. What is triplet loss and why is it needed?

### Answer:


### Q6. What is neural style transfer (NST) and how does it work?

### Answer:


### Q7. Describe style cost function.

### Answer:
