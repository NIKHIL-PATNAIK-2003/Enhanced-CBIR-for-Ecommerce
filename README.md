# **Enhanced CBIR System for Commercial Applications**

🚀 **A Deep Learning-Based Content-Based Image Retrieval (CBIR) System for E-Commerce**  

## 🔍 Overview  
This repository contains an **Enhanced CBIR System** designed for fashion products using deep learning models. The system allows users to search for products based on images instead of text-based queries. It utilizes **ResNet-50, VGG16, and a Hybrid Model (VGG16 + EfficientNetB0 + MobileNetV3Small)** for feature extraction and retrieval.

## 📌 Features  
- **Image-Based Product Search**: Users can upload an image to retrieve visually similar products.  
- **Deep Learning Models**: Implements ResNet-50, VGG16, and a Hybrid Model to extract features from images.  
- **Cosine Similarity Matching**: Ranks images based on similarity scores.  
- **Scalable & Efficient**: Designed for large e-commerce datasets.  
- **Visualization Tools**: Includes ROC curves, confusion matrices, and retrieval result visualization.  

## 📂 Repository Structure  
```
/CBIR_Project
│── hybrid.ipynb                  # Hybrid model implementation (VGG16 + EfficientNetB0 + MobileNetV3Small)
│── VGG16.ipynb                    # VGG16 model implementation for feature extraction
│── ResNet50.py                    # Python script implementing ResNet-50-based CBIR
│── final-major-report.docx         # Detailed project report
│── CBIR_Major Final.pptx           # Project presentation slides
│── dataset/                        # Folder to store processed images (Fashion MNIST or custom dataset)
│── results/                        # Directory for storing retrieval outputs and plots
│── README.md                       # Project documentation
```

## 📊 Models Used  
1. **ResNet-50**  
   - Pretrained on ImageNet, fine-tuned for fashion image retrieval.  
2. **VGG16**  
   - Extracts features from images and retrieves similar products.  
3. **Hybrid Model (VGG16 + EfficientNetB0 + MobileNetV3Small)**  
   - Combines three deep learning models for improved feature extraction and retrieval accuracy.  

## 🏆 Achievements  
- Finalist in **Smart India Hackathon 2024**  
- Finalist in **Dark Patterns Buster Hackathon** (Top 150 in India)  
- Selected for **Google Solution Challenge Bootcamp**  
- Prefinalist in **Harvest Innovation Hackathon**  

## 🔧 Installation & Setup  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/CBIR_Project.git
   cd CBIR_Project
   ```
2. **Install Dependencies**  
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn seaborn pillow
   ```
3. **Run ResNet-50 CBIR System**  
   ```bash
   python ResNet50.py
   ```
4. **Use Jupyter Notebooks for Hybrid & VGG16 Models**  
   ```bash
   jupyter notebook
   ```

## 🖼️ How It Works  
1. Upload an image as a query.  
2. The system extracts features using the deep learning model.  
3. The extracted features are compared with a database of product images using cosine similarity.  
4. The most similar images are retrieved and displayed.  

## 📈 Evaluation  
- **Accuracy Metrics**: F1-score, Precision, Recall  
- **Visualization**: Confusion Matrix, ROC Curve  
- **Performance Comparison**: ResNet-50 vs. VGG16 vs. Hybrid Model  

## 🔮 Future Improvements  
- Support for **real-time product search** in e-commerce.  
- Expansion to **high-resolution, multi-category datasets**.  
- **Personalized recommendations** based on user behavior.  
- **Faster search retrieval** with improved indexing techniques.  

## 📜 Citation  
If you use this project for research or academic purposes, please cite:  
📄 **[Final Report](final-major-report.docx)** | 📊 **[Presentation Slides](CBIR_Major Final.pptx)**  
