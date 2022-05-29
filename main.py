
from flask import Flask,request,jsonify
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import association_rules,apriori, fpmax, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pickle
import json

app = Flask(__name__)

@app.route("/test",methods=['GET'])
def predict():
    if request.method == 'GET':
        inp = request.json
        name = inp['services'] 
        # print(name)
        # print(type(name))
        pickle_in = open("D:\\model.pickle","rb")
        rules = pickle.load(pickle_in)
        res = {}
        x=rules[ rules['antecedents'] == set(name) ]['consequents']
        for i in x:
            brk = list(i)
            for j in range(len(brk)):
                if brk[j]!= 'nan':
                    res[brk[j]] = 1
                
        print(json.dumps(list(res.keys())))

        return jsonify({"response":json.dumps(list(res.keys()))})

@app.route("/train",methods=['GET'])
def train():
    if request.method == 'GET':
        Data = pd.read_csv('data2.csv', header = None)
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
        pickle.dump(rules, open("C:\\Users\\Owais Sultan\\Downloads\\output\\test_api\\model.pickle", "wb"))
        return jsonify({"response":"done"})






if __name__ == '__main__':
    app.run()