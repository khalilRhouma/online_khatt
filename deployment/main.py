import logging

from flask import Flask, render_template, request, session

from inference import do_inference

app = Flask(__name__)
app.config.from_envvar("FLASK_APPLICATION_SETTINGS")
app.secret_key = "online_khatt"
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


@app.route("/")
def index():
    session.clear()
    title = "Online Khatt"
    return render_template("index.html", title=title)


@app.route("/add-data", methods=["GET"])
def add_data_get():
    # message = session.get('message', '')
    return render_template("collect_data.html", points="")


@app.route("/add-data", methods=["POST"])
def add_data_post():

    # label = request.form['letter']
    # labels = np.load('data/labels.npy')
    # labels = np.append(labels, label)
    # np.save('data/labels.npy', labels)

    # pixels = request.form['pixels']
    # pixels = pixels.split(',')
    # img = np.array(pixels).astype(float).reshape(1, 50, 50)
    # imgs = np.load('data/images.npy')
    # imgs = np.vstack([imgs, img])
    # np.save('data/images.npy', imgs)

    # session['message'] = f'"{label}" added to the training dataset'
    points = eval(f"[{request.form['points']}]")
    points = [points[i : i + 3] for i in range(0, len(points), 3)]
    # return redirect(url_for('add_data_get'))
    return render_template("collect_data.html", points=points)


@app.route("/prediction", methods=["GET"])
def prediction_get():

    return render_template("prediction.html", prediction="")


@app.route("/prediction", methods=["POST"])
def prediction_post():
    try:
        # letter = request.form['letter']
        if request.form["points"] != "":
            points = eval(f"[{request.form['points']}]")
            points = [points[i : i + 3] for i in range(0, len(points), 3)]
            prediction = do_inference(
                points,
                config_file=app.config["CONFIG_FILE"],
                model_path=app.config["MODLE_PATH"],
                lm_binary_path=app.config["LM_BINARY_PATH"],
                lm_trie_path=app.config["LM_TRIE_PATH"],
            )
        else:
            prediction = ""
        return render_template("prediction.html", prediction=prediction)

    except Exception as e:
        print(e)
        return render_template("error.html")


if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    # app.run(debug=True, host="0.0.0.0", port=port)
    app.run()
