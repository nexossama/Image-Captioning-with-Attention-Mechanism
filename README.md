# Image Captioning with Attention

## Project Overview

This project is an innovative AI-powered storytelling application that transforms your images into captivating narratives. The application works through a two-step process:

1. **Image Captioning with Attention**:

   - We use a pre-trained Image Captioning model with an Attention mechanism to generate detailed, descriptive captions for each input image.
   - The attention mechanism allows the model to focus on the most relevant parts of the image when generating captions.

2. **Story Generation with Gemini**:
   - The generated image captions are then sent to Google's Gemini AI.
   - You can specify your preferred storytelling style (e.g., funny, sad, horror, etc.).
   - Gemini uses the image captions to craft a unique, style-specific story that brings your images to life.

## Project Demo
https://github.com/user-attachments/assets/cbec7bca-e59c-463f-8409-f2c008d39f93

## Project architecture

- System architecture

![alt text](Readme-assets/Project%20architecture.png)

- Image captioning model architecture

![alt text](Readme-assets/image%20captioning.png)


The attention mechanism combines short and long-term memory to create a more intelligent understanding of the image and its context, allowing the model to remember and reference important details throughout the entire caption generation process.

for more details about the building process of the model ,you can check [Image-captioning-with-Attention.ipynb](assets/Image-captioning-with-Attention.ipynb) file in assets folder

## Prerequisites

- Anaconda or Miniconda
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Image-Captioning-with-Attention.git
cd Image-Captioning-with-Attention
```

### 2. Create Conda Environment

```bash
conda create -n image-caption python=3.10 -y
conda activate image-caption
```

### 3. Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Open the .env file and fill in the required variables
# You may need to add:
# - API keys
# - Model paths
# - Other configuration settings
nano .env
```

### 5. Run Backend Server

```bash
# Ensure you are in the src directory
cd src

# Start the FastAPI backend server
uvicorn main:app --host 0.0.0.0 --port 5000
```

### 6. Open Frontend

- Open `index.html` in your web browser to use the application

## Troubleshooting

- Ensure all dependencies are correctly installed
- Check that your `.env` file is properly configured
- Verify that you have the required model weights and datasets

<!-- ## Demo

[![Demo Video](https://img.shields.io/badge/Watch-Demo-blue?style=for-the-badge)](Readme-assets/Bed_time_story.m4v) -->

## Contributors

Thanks to the amazing people who have contributed to this project! 💪🚀

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/aymane-maghouti">
        <img src="https://avatars.githubusercontent.com/aymane-maghouti" width="80px;" alt="Contributor 2"/>
        <br/>
        <sub><b>Aymane Maghouti</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/nexossama">
        <img src="https://avatars.githubusercontent.com/nexossama" width="80px;" alt="Colleague 3"/> <br/>
        <sub><b>Ossama Outmani</b></sub>
      </a>
  </tr>
</table>

## Contact

If you have any questions or suggestions, feel free to reach out to any of us!

### Team Members

**Name**: Aymane Maghouti  
**Email**: [aymanemaghouti16@gmail.com](mailto:aymanemaghouti16@gmail.com)  
**LinkedIn**: [Aymane's LinkedIn](https://www.linkedin.com/in/aymane-maghouti/)

**Name**: Ossama Outmani  
**Email**: [ossamaoutmani@gmail.com](mailto:ossamaoutmani@gmail.com)  
**LinkedIn**: [Ossama's LinkedIn](https://www.linkedin.com/in/ossama-outmani/)
