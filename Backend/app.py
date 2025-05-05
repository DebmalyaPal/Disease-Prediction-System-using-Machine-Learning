from flask import Flask

app = Flask(__name__)

symptoms = {}

def load_symptoms():
    global symptoms
    symptom_list = []
    try:
        with open('./Data/Training.csv', 'r') as file:
            contents = file.readline()
            for raw_symptom in contents.strip().split(','):
                if len(raw_symptom):
                    symptom = raw_symptom.replace('_', ' ').title()
                    symptom_list.append(symptom)
            symptom_list.sort()
            print(symptom_list)
    except FileNotFoundError:
        print(f"Error: The file was not found.")
    except PermissionError:
        print(f"Error: You do not have permission to read.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    symptoms = {index: symptom for index, symptom in enumerate(symptom_list)}

@app.before_first_request
def startup():
    load_symptoms()

@app.route("/symptoms", methods=["GET"])
def get_all_symptoms():
    return symptoms

@app.route("/predict", methods=["POST"])
def predict():
    symptom_list=[]
    disease_list=[]
    return disease_list
