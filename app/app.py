from flask import Flask, render_template
from flask_ngrok import run_with_ngrok

app   = Flask(__name__, static_url_path='/static')

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def index():
    return render_template('index.html')

# [Routing untuk Halaman About]
@app.route("/about")
def about():
    return render_template('about.html')

# [Routing untuk Halaman apikasi]
@app.route("/aplikasi")
def aplikasi():
    return render_template('aplikasi.html')

# [Routing untuk Halaman team]
@app.route("/team")
def team():
    return render_template('team.html')


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)

