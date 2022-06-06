import random
from flask import Flask,request,jsonify
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pickle
import json

app = Flask(__name__)

@app.route("/test",methods=['POST'])
def predict():
    services = {
                "1":"roofing services",
                "2":"flooring services",
                "3":"landscaping services",
                "4":"electrical services",
                "5":"junk removal",
                "6":"moving services",
                "7":"masonry services",
                "8":"appliance services",
                "9":"plumbing services",
                "10":"locksmith services",
                "11":"hvac services",
                "12":"pest control",
                "13":"painting services",
                "14":"towing services",
                "15":"vehicle repair"

    }

    if request.method == 'POST':
        inp = request.json
        name = inp['services'] 
        # print(name)
        # print(type(name))
        pickle_in = open("model.pickle","rb")
        rules = pickle.load(pickle_in)
        res = {}
        x=rules[ rules['antecedents'] == set(name) ]['consequents']
        for i in x:
            brk = list(i)
            for j in range(len(brk)):
                if brk[j]!= 'nan':
                    res[brk[j]] = 1
                
        # print(json.dumps(list(res.keys())))

        if bool(res) == False:
            for i in range(0,9):
             res[services[str(random.randint(1,15))]] = str(i)

        return jsonify({"response":json.dumps(list(res.keys()))})

@app.route("/train",methods=['GET'])
def train():
    if request.method == 'GET':
        Data = pd.read_csv('newdata.csv', header = None)
        Data.drop(columns=Data.columns[:1], 
                axis=1, 
                inplace=True)
        Data = Data.iloc[1: , :]

        dataset = []
        # populating a list of transactions
        for i in range(0, 1000): 
            dataset.append([str(Data.values[i,j]) for j in range(0, 12)])
            dataset

        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = fpgrowth(df, min_support=0.02, use_colnames=True)    
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.9)
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        pickle.dump(rules, open("model.pickle", "wb"))
        return jsonify({"response":"done"})






if __name__ == '__main__':
    app.run()