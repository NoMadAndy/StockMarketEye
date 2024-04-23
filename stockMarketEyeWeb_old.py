from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import io
import base64
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

class InputForm(FlaskForm):
    symbol = StringField('Aktiensymbol', validators=[DataRequired()])
    start_date = StringField('Startdatum (YYYY-MM-DD)', default='2017-01-01', validators=[DataRequired()])
    end_date = StringField('Enddatum (YYYY-MM-DD)', default=datetime.now().strftime('%Y-%m-%d'), validators=[DataRequired()])
    submit = SubmitField('Vorhersage')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    plot_url = None
    debug_logs = []
    
    if form.validate_on_submit():
        debug_logs.append("Download der Daten gestartet.")
        stock_data = yf.download(form.symbol.data, start=form.start_date.data, end=form.end_date.data)
        debug_logs.append("Daten erfolgreich heruntergeladen.")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        debug_logs.append("Daten skalieren.")
        scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
        debug_logs.append("Daten erfolgreich skaliert.")

        def create_dataset(data, time_step=100):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)
        
        X, y = create_dataset(scaled_data, 100)
        debug_logs.append("Dataset erstellt.")
        
        train_size = 0.8
        X_train, X_test = X[:int(X.shape[0]*train_size)], X[int(X.shape[0]*train_size):]
        y_train, y_test = y[:int(y.shape[0]*train_size)], y[int(y.shape[0]*train_size):]
        
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=(100, 1)),
            LSTM(units=64),
            Dense(units=64),
            Dense(units=1)
        ])
        debug_logs.append("Modell wird kompiliert und trainiert.")
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64)
        debug_logs.append("Modelltraining abgeschlossen.")
        
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        debug_logs.append("Vorhersagen wurden gemacht und transformiert.")
        
        original_data = stock_data['Close'].values
        predicted_data = np.empty_like(original_data)
        predicted_data[:] = np.nan
        predicted_data[-len(predictions):] = predictions.reshape(-1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(original_data, label='Original Data')
        plt.plot(predicted_data, label='Predicted Data')
        plt.legend()
        plt.title('Stock Price Prediction')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', form=form, plot_url=plot_url, debug_logs=debug_logs)

if __name__ == '__main__':
    app.run(debug=True)