from flask import Flask, jsonify, request, send_file
from it.valtellina.eda import PuliziaDatasetFunghi
from it.valtellina.logistic_regression import RegressioneLogistica
from it.valtellina.svc import SupportVectorMachine

app = Flask(__name__)

@app.route('/na')
def individua_na():
    df = PuliziaDatasetFunghi()
    null, perc = df.individua_na()
    return jsonify({
        "null values": null.to_dict(),
        "percentage": perc.to_dict()
    })


@app.route('/grafici')
def grafici():
    df = PuliziaDatasetFunghi()
    img = df.grafici_distribuzioni()
    return send_file(img, mimetype='image/png')

@app.route('/stats')
def stats():
    df = PuliziaDatasetFunghi()
    stats = df.statistica_descrittiva()
    return jsonify(stats)



@app.route('/logreg/coeff')
def logistic_coeff():
    df = PuliziaDatasetFunghi()
    X, y = df.elimina_na()
    model = RegressioneLogistica(X, y)
    model.train()
    img = model.grafico_coefficienti()
    return send_file(img, mimetype='image/png')

@app.route('/logreg/confusionmatrix')
def logistic_confusionmatrix():
    df = PuliziaDatasetFunghi()
    X, y = df.elimina_na()
    model = RegressioneLogistica(X, y)
    model.train()
    img = model.grafico_matrice_confusione()
    return send_file(img, mimetype='image/png')

@app.route('/logreg/summary')
def logistic_summary():
    df = PuliziaDatasetFunghi()
    X, y = df.elimina_na()
    model = RegressioneLogistica(X, y)
    model.train()
    summary = model.summary()
    return jsonify(summary)




@app.route('/svc/confusionmatrix')
def svm_confusionmatrix():
    df = PuliziaDatasetFunghi()
    X, y = df.elimina_na()
    model = SupportVectorMachine(X, y)
    model.train()
    img = model.grafico_matrice_confusione()
    return send_file(img, mimetype='image/png')

@app.route('/svc/summary')
def svm_summary():
    df = PuliziaDatasetFunghi()
    X, y = df.elimina_na()
    model = SupportVectorMachine(X, y)
    model.train()
    summary = model.summary()
    return jsonify(summary)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)