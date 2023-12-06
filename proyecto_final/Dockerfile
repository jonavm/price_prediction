# Utiliza una imagen base oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Establece variables de entorno necesarias para Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Instalar Rust y Cargo
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env

# Actualizar pip
RUN pip install --upgrade pip

# Copia el archivo requirements.txt en el contenedor
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu aplicación en el contenedor
COPY . .

# Expone el puerto 5000
EXPOSE 5000

# Crea un usuario no-root para ejecutar la aplicación
RUN useradd -m myuser
USER myuser

# El comando para ejecutar la aplicación
CMD ["flask", "run"]
