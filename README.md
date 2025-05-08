### README

# RAG System Prototype

This project is a **Retrieval-Augmented Generation (RAG)** system prototype built using **Streamlit**. The app allows users to upload documents, query them, and retrieve results based on selected translation and embedding methods.

---

## Features
- **Interactive Sidebar Settings**:
  - Choose from multiple translation methods: `Easy`, `Mbart50`, `MarianMT`, `T5`, `Pegasus`, `Google`.
  - Select embedding methods: `SBERT`, `BERT`, `FastText`, `GPT-3`, `Word2Vec`.
  - Configure the number of results (`Top K`) to display.

- **Document Upload**:
  - Upload multiple documents (PDF or TXT) simultaneously.
  - View uploaded files with file size information.
  - Delete selected documents from the session.

- **Query System**:
  - Enter a query and retrieve results based on the selected embedding method.
  - Display a configurable number of results (`Top K`) with placeholder text.

- **Session Management**:
  - Clear all session data using the `Clear All Data` button.
  - Dynamic keys ensure file uploaders reset after each upload.

- **Streamlit Layout**:
  - Wide layout for a seamless user experience.
  - Organized sections for easy navigation.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install streamlit
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Start the App**: Open the app in your browser after running the `streamlit run` command.
2. **Configure Settings**:
   - Use the sidebar to select translation and embedding methods.
   - Adjust the number of results to display.
3. **Upload Documents**:
   - Drag and drop or select files in the "Document Upload" section.
   - View or delete uploaded documents as needed.
4. **Query the System**:
   - Enter a query in the "Query Input" section.
   - View the retrieved results based on the selected embedding method.

---

## Future Enhancements
- Integrate real translation and embedding methods.
- Implement a database or cloud storage for uploaded documents.
- Improve result generation with advanced RAG algorithms.
- Add support for additional document formats.

---

## License
This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgments
Built with ❤️ using [Streamlit](https://streamlit.io/) for prototyping RAG systems.