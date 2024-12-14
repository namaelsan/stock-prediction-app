from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import pyodbc
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 's3cr3t'

# Veritabanı bağlantısı
def get_db_connection():
    conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                          f'SERVER=localhost;'
                          f'DATABASE=fintek_db;'
                          f'UID=sa;'
                          f'PWD=Mei@2024!')
    return conn

def read_csv_data(company_name):
    file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Veri Toplama', 'data')), f'{company_name}.csv')
    
    print(f"Aranan dosya yolu: {file_path}")  # Dosya yolunu yazdırmak için
    
    try:
        if not os.path.exists(file_path):
            print(f"{file_path} dosyası bulunamadı.")
            return None

        # CSV dosyasını okuma işlemi
        df = pd.read_csv(file_path, header=0)  # header=1 diyerek 1. satırı başlık olarak alıyoruz

        # Sütunları yeniden adlandır (örnek veriye uygun olarak)
        df.columns = ['date', 'close', 'volume']

        # Son satıra git
        last_row = df.iloc[-1]

        # Son satırdaki verileri al
        date = last_row['date']
        close_price = last_row['close']
        volume = last_row['volume']
        
        print(f"Son satırdaki veri:\nTarih: {date}, Kapanış Fiyatı: {close_price}, Hacim: {volume}")
        
        return df
    except FileNotFoundError:
        print(f"{file_path} dosyası bulunamadı.")
        return None
    except Exception as e:
        print(f"CSV okuma hatası: {e}")
        return None


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM Kullanicilar WHERE eposta = ?', (email,))
            user = cursor.fetchone()

            if not user:
                flash('E-posta adresi bulunamadı.', 'error')
                return render_template('login.html')

            print(f"Veritabanından gelen şifre: {user[5]}")  # Hashli şifreyi yazdıralım
            if check_password_hash(user[5], password):
                session['user_id'] = user[0]
                return redirect(url_for('home'))
            else:
                flash('Şifreyi kontrol edin', 'error')

        except Exception as e:
            flash(f'Giriş sırasında bir hata oluştu: {str(e)}', 'error')

        finally:
            conn.close()

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']

        if not (first_name and last_name and email and password and phone):
            flash('Boş alanları doldurunuz.', 'error')
            return render_template('signup.html')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Kullanicilar WHERE eposta = ?', (email,))
        existing_user = cursor.fetchone()
        conn.close()

        if existing_user:
            flash('Bu e-posta ile zaten bir hesap bulunmaktadır.', 'error')
            return render_template('signup.html')

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO Kullanicilar (ad, soyad, eposta, telefon, sifre)
                VALUES (?, ?, ?, ?, ?)
            ''', (first_name, last_name, email, phone, hashed_password))

            conn.commit()

        except Exception as e:
            conn.rollback()
            flash('Kayıt sırasında bir hata oluştu.', 'error')

        finally:
            conn.close()

        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Kullanicilar WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()

    cursor.execute('SELECT * FROM Bist100Sirketleri')
    companies = cursor.fetchall()

    conn.close()

    return render_template('home.html', user=user, companies=companies)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    company_id = request.args.get('company_id')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Bist100Sirketleri WHERE id = ?', (company_id,))
    company = cursor.fetchone()

    company_name = company[1]  # Şirket adını alıyoruz (örneğin: company[1] 'Şirket Adı')

    # CSV dosyasındaki veriyi okuma
    csv_data = read_csv_data(company_name)
    if csv_data is not None:
        # Son günün tarihini, hacmini ve kapanış değerini alıyoruz
        last_row = csv_data.iloc[-1]  # Son satırı al
        date = last_row['date']  # 'Price' sütunu aslında 'Date'
        close_price = last_row['close']  # 'Close' sütunu
        volume = last_row['volume']  # 'Volume' sütunu

        # Tahmin fiyatını dosyadan okuma (company_name ile eşleşen satır)
        predicted_price = None
        try:
            # Üst klasördeki dosyaya erişmek için doğru yolu kullanıyoruz
            file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'predicted_stock_prices.txt')
            with open(file_path, 'r') as file:
                for line in file:
                    # Her satırda şirket adı ve tahmin fiyatı olduğunu varsayıyoruz
                    # Şirket adı ve tahmin fiyatı arasındaki ayırıcıyı (örneğin: virgül) kullanıyoruz
                    parts = line.strip().split('\t')
                    if len(parts) == 2 and parts[0] == company_name:
                        predicted_price = float(parts[1])
                        break  # Eşleşen satırı bulduğumuzda çıkıyoruz
        except FileNotFoundError:
            print("predicted_stock_prices.txt dosyası bulunamadı.")

        # Tahmin fiyatını şablona gönder
        return render_template('prediction.html', company=company, 
                               date=date, volume=volume, close_price=close_price, 
                               company_name=company_name, predicted_price=predicted_price)

    conn.close()

    return render_template('prediction.html', company=company, csv_data=None)



@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
