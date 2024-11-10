# IELTS Bands Master

**First IELTS Writing & Speaking Test with AI Feedback**

<img src="IELTS BANDS.png" >

# Description

**IELTS Bands Master** is a web application designed to help you practice and enhance your IELTS Writing and Speaking skills. By simulating the IELTS test, the app provides real-time feedback based on IELTS scoring criteria. The app uses AI to analyze your responses and gives actionable insights to help you improve your fluency, vocabulary, grammar, and overall band score.

## Key Features

- **AI-Powered Feedback**: Get personalized feedback for both Writing and Speaking tasks.
- **Writing Test Simulation**: Includes Writing Task 1 (graph/chart description) and Writing Task 2 (essay).
- **Speaking Test Simulation**: Simulates Part 1 (Introduction), Part 2 (Long Turn), and Part 3 (Discussion).
- **Speech-to-Text**: Take the Speaking test using speech-to-text functionality for more realistic practice.
- **Band Score Estimation**: Receive an estimated band score based on your responses to help track your progress.

## How to Use

1. **Choose Your Test Type**: Select between Writing or Speaking test from the sidebar.
2. **Complete the Tasks**: Write your responses or speak into your microphone for the Speaking test.
3. **Receive AI Feedback**: After completing your test, get feedback on fluency, vocabulary, grammar, and more.
4. **Track Your Progress**: Use the app to track improvements and work towards your target band score.

## Upcoming Features

- **Mock Tests**: Full IELTS test simulations to mimic real exam conditions.
- **Progress Tracking**: Detailed insights into your performance over time.
- **More AI Scoring Improvements**: Enhanced AI models for more accurate feedback.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Pip (Python package manager)

### Installation

## Clone the repository:

````bash
git https://github.com/umairazmat/IELTS-Bands-Master
cd IELTS Bands Master

## Set up a virtual environment:

```bash
python -m venv IELTS-Bands-Master
````

## For Windows:

```bash
.\IELTS-Bands-Master\Scripts\activate
```

## For Mac/Linux:

```bash
source IELTS-Bands-Master/bin/activate
```

## Install the required packages:

```bash
pip install -r requirements.txt
```

## Create a .env file in the root directory with the following content:

```bash
OPENAI_API_KEY=your_actual_api_key
OPENAI_API_BASE=your_actual_api_key
GROQ_API_KEY=your_actual_api_key
```

## Run the application:

```bash
streamlit run .\app.py
```

## if script not run :

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Troubleshooting

- Verify that API keys are correctly set in the `.env` file.
- Ensure that the Python version and dependencies match the requirements.

---

## Contributing

1. **Fork the repository**
2. **Create a feature branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

   3- **Commit your changes**

   ```bash
   git commit -m 'Add new feature'
   ```

4- **Push to the branch**
`bash
git push origin feature/YourFeature
    `bash

5- **Open a Pull Request**

## 📞 Contact

For queries or collaboration, please reach out at **hello@umairazmat.com**.
