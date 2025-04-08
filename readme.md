
# EEGApp

**EEGApp** is an application designed for the processing and analysis of electroencephalography (EEG) data. It leverages web technologies and Python libraries to provide an interactive interface and advanced analysis tools.

## Features

- **EEG Signal Processing**: Implements techniques for cleaning and filtering EEG signals.
- **Interactive Visualization**: Offers graphs and visual representations of the processed EEG data.
- **Web Interface**: Developed with modern web technologies for ease of use.

## Project Structure

- **`processing/`**: Contains scripts and modules for EEG signal processing.
- **`static/`**: Static files such as stylesheets, JavaScript scripts, and other resources.
- **`templates/`**: HTML templates for the user interface.
- **`app.py`**: Main file that launches the web application.
- **`requirements.txt`**: List of dependencies required to run the application.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/marioquiles/EEGApp.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd EEGApp
   ```

3. **Create and activate a virtual environment (optional but recommended)**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**:

   ```bash
   python app.py
   ```

2. **Access the web interface**: Open a browser and go to `http://localhost:5000` to interact with the application.

## Contributions

Contributions are welcome. If you wish to collaborate, please follow these steps:

1. **Fork** this repository.
2. **Create a branch** for your new feature (`git checkout -b feature/new-feature`).
3. **Make your changes** and commit them (`git commit -am 'Add new feature'`).
4. **Push your changes** to your fork (`git push origin feature/new-feature`).
5. **Open a Pull Request** in this repository.

## License

This project is distributed under the MIT license. See the `LICENSE` file for more details.
