from flask import Flask, render_template, request
import model as m

app = Flask(__name__,template_folder='template')


# @app.route("/",methods=['POST'])
@app.route("/",methods=["GET","POST"])
def hello():
    mks=None
    mkss=None
    mksss=None

    if request.method == "POST":
        Temp = request.form['Temp']
        Press = request.form['Press']
        Rain = request.form['Rain']
        Wind = request.form['Wind']
        pred = m.prediction(Temp,Press,Rain,Wind)
        # print(pred)
        # return render_template("index.html")
        mks = pred
        predd = m.ypredict()
        preddd= m.mss()
        mkss=predd
        mksss=preddd
    return render_template('index.html', p=mks,pp=mkss,ppp=mksss)


# @app.route("/sub",methods=['POST'])
# def submit():
#     if request.method =="POST":
#         name =request.form["username"]
#     return render_template("submit.html",n=name)
if __name__=="__main__":
    app.run(debug=True)