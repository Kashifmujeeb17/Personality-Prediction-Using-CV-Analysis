import os
import pandas as pd
from tkinter import *
from tkinter import filedialog
import tkinter.font as font
import nltk
nltk.download('stopwords')
import en_core_web_sm
import textract
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore", message="[W094]")


class train_model:
    def train(self):
        data = pd.read_csv('U:/pythonProject1/training_dataset.csv')
        array = data.values

        for i in range(len(array)):
            if array[i][0] == "Male":
                array[i][0] = 1
            else:
                array[i][0] = 0

        df = pd.DataFrame(array)

        maindf = df[[0, 1, 2, 3, 4, 5, 6]]
        mainarray = maindf.values

        temp = df[7]
        train_y = temp.values

        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.mul_lr.fit(mainarray, train_y)

    def test(self, test_data):
        try:
            test_predict = list()
            for i in test_data:
                test_predict.append(int(i))
            y_pred = self.mul_lr.predict([test_predict])
            return y_pred
        except:
            print("All Factors For Finding Personality Not Entered!")


def check_type(data):
    if isinstance(data, str):
        return str(data).title()
    if isinstance(data, (list, tuple)):
        str_list = ""
        for i, item in enumerate(data):
            str_list += item + ", "
        return str_list
    else:
        return str(data)


import re

def extract_name(text):
    # Extracts the name from the resume text using regular expressions
    # Modify this logic based on the structure and format of your resumes
    name_regex = r"Name:\s*(\w+\s*\w+)"
    match = re.search(name_regex, text)
    if match:
        return match.group(1)
    else:
        return None



def extract_phone_number(text):
    # Use regular expression pattern to search for phone number
    phone_number_pattern = r"\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    phone_number_matches = re.findall(phone_number_pattern, text)
    if phone_number_matches:
        return phone_number_matches[0]
    else:
        return None

def extract_email(text):
    # Use regular expression pattern to search for email address
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    email_matches = re.findall(email_pattern, text)
    if email_matches:
        return email_matches[0]
    else:
        return None


def prediction_result(top, aplcnt_name, cv_path, personality_values):
    "after applying for a job"
    top.withdraw()
    applicant_data = {"Candidate Name": aplcnt_name.get(), "CV Location": cv_path}

    age = personality_values[1]

    print("\n############# Candidate Entered Data #############\n")
    print(applicant_data, personality_values)

    personality = model.test(personality_values)
    print("\n############# Predicted Personality #############\n")
    print(personality)

    text = textract.process(cv_path).decode('utf-8')  # Extract text from the document

    data = {}  # Dictionary to store extracted information from the resume

    # Extracting relevant information from the resume text
    # Modify this section based on the structure and format of your resumes
    # Example: data['name'] = extract_name(text)
    data['name'] = extract_name(text)
    data['mobile_number'] = extract_phone_number(text)
    data['email'] = extract_email(text)
    # Extract more information as needed

    nlp = en_core_web_sm.load()
    del data['name']
    if data.get('mobile_number') is not None and len(data.get('mobile_number', '')) < 10:
        del data['mobile_number']

    print("\n############# Resume Parsed Data #############\n")
    print(data)

    result = Tk()
    result.overrideredirect(False)
    result.geometry("{0}x{1}+0+0".format(result.winfo_screenwidth(), result.winfo_screenheight()))
    result.configure(background='White')
    result.title("Predicted Personality")

    # Title
    titleFont = font.Font(family='Arial', size=40, weight='bold')
    Label(result, text="Result - Personality Prediction", foreground='green', bg='white', font=titleFont, pady=10,
          anchor=CENTER).pack(fill=BOTH)

    Label(result, text=str('{}: {}'.format("Name", aplcnt_name.get())).title(), foreground='black', bg='white',
          anchor='w').pack(fill=BOTH)
    Label(result, text=str('{}: {}'.format("Age", age)), foreground='black', bg='white', anchor='w').pack(fill=BOTH)
    for key, value in data.items():
        if value is not None:
            Label(result, text=str('{}: {}'.format(check_type(key.title()), check_type(value))),
                  foreground='black', bg='white', anchor='w', width=60).pack(fill=BOTH)
    Label(result, text=str("Predicted Personality: " + personality).title(), foreground='black', bg='white',
          anchor='w').pack(fill=BOTH)

    quitBtn = Button(result, text="Exit", command=result.destroy).pack()

    terms_mean = """
# Openness:
    People who like to learn new things and enjoy new experiences usually score high in openness. Openness includes traits like being insightful and imaginative and having a wide variety of interests.

# Conscientiousness:
    People that have a high degree of conscientiousness are reliable and prompt. Traits include being organized, methodic, and thorough.

# Extraversion:
    Extraversion traits include being energetic, talkative, and assertive. Extraverts get their energy and drive from others, while introverts are self-driven and get their drive from within themselves.

# Agreeableness:
    Individuals with high agreeableness are warm, friendly, compassionate, and cooperative. Traits include being kind, affectionate, and sympathetic. In contrast, people with lower levels of agreeableness may be more distant.

# Neuroticism:
    Neuroticism or Emotional Stability relates to the degree of negative emotions. People that score high on neuroticism often experience emotional instability and negative emotions. Characteristics typically include being moody and tense.    
"""

    Label(result, text=terms_mean, foreground='green', bg='white', anchor='w', justify=LEFT).pack(fill=BOTH)

    result.mainloop()


def predict_personality():
    """Predict Personality"""

    # Closing The Previous Window
    root.withdraw()

    # Creating new window
    top = Toplevel()
    top.geometry('700x500')
    top.configure(background='black')
    top.title("Apply For A Job")

    # Title
    titleFont = font.Font(family='Helvetica', size=20, weight='bold')
    lab = Label(top, text="Personality Prediction", foreground='red', bg='black', font=titleFont, pady=10).pack()

    # Job_Form
    job_list = ('Select Job', '101-Developer at TTC', '102-Chef at Taj', '103-Professor at MIT')
    job = StringVar(top)
    job.set(job_list[0])

    Label(top, text="Applicant Name", foreground='white', bg='black').place(x=70, y=130)
    Label(top, text="Age", foreground='white', bg='black').place(x=70, y=160)
    Label(top, text="Gender", foreground='white', bg='black').place(x=70, y=190)
    Label(top, text="Upload Resume", foreground='white', bg='black').place(x=70, y=220)
    Label(top, text="Enjoy New Experience or thing (Openness)", foreground='white', bg='black').place(x=70, y=250)
    Label(top, text="How Often You Feel Negativity (Neuroticism)", foreground='white', bg='black').place(x=70, y=280)
    Label(top, text="Wishing to do one's work well and thoroughly (Conscientiousness)", foreground='white',
          bg='black').place(x=70, y=310)
    Label(top, text="How much would you like to work with your peers (Agreeableness)", foreground='white',
          bg='black').place(x=70, y=340)
    Label(top, text="How outgoing and social interaction you like (Extraversion)", foreground='white',
          bg='black').place(x=70, y=370)

    sName = Entry(top)
    sName.place(x=450, y=130, width=160)
    age = Entry(top)
    age.place(x=450, y=160, width=160)
    gender = IntVar()
    R1 = Radiobutton(top, text="Male", variable=gender, value=1, padx=7)
    R1.place(x=450, y=190)
    R2 = Radiobutton(top, text="Female", variable=gender, value=0, padx=3)
    R2.place(x=540, y=190)
    cv = Button(top, text="Select File", command=lambda: OpenFile(cv))
    cv.place(x=450, y=220, width=160)
    openness = Scale(top, from_=1, to=10, orient=HORIZONTAL)
    openness.place(x=450, y=250, width=160)

    neuroticism = Scale(top, from_=1, to=10, orient=HORIZONTAL)
    neuroticism.place(x=450, y=280, width=160)

    conscientiousness = Scale(top, from_=1, to=10, orient=HORIZONTAL)
    conscientiousness.place(x=450, y=310, width=160)

    agreeableness = Scale(top, from_=1, to=10, orient=HORIZONTAL)
    agreeableness.place(x=450, y=340, width=160)

    extraversion = Scale(top, from_=1, to=10, orient=HORIZONTAL)
    extraversion.place(x=450, y=370, width=160)

    submitBtn = Button(top, padx=2, pady=0, text="Submit", bd=0, foreground='white', bg='red', font=(12))
    submitBtn.config(command=lambda: prediction_result(top, sName, loc, (
        gender.get(), age.get(), openness.get(), neuroticism.get(), conscientiousness.get(), agreeableness.get(),
        extraversion.get())))
    submitBtn.place(x=350, y=430, width=200)

    top.mainloop()


def OpenFile(b4):
    global loc;
    name = filedialog.askopenfilename(initialdir="C:/Users/Mr.Kashif/Downloads/Compressed/project",
                                      filetypes=(("Document", "*.docx*"), ("PDF", "*.pdf*"), ('All files', '*')),
                                      title="Choose a file."
                                      )
    try:
        filename = os.path.basename(name)
        loc = name
    except:
        filename = name
        loc = name
    b4.config(text=filename)
    return


if __name__ == "__main__":
    model = train_model()
    model.train()

    root = Tk()
    root.geometry('700x500')
    root.configure(background='white')
    root.title("Personality Prediction System")
    titleFont = font.Font(family='Arial', size=20, weight='bold')
    Label(root, text="Personality Prediction System", foreground='red', bg='white', font=titleFont, pady=10).pack()
    btn = Button(root, text="Click Here To Apply For A Job", padx=50, pady=10, bd=0, bg='red', foreground='white',
                 font=(12))
    btn.config(command=predict_personality)
    btn.pack(pady=100)
    root.mainloop()