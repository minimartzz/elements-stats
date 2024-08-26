<div align="center">

  <img src="assets/header.png" alt="logo" width="400" height="auto" />
  <h1>Elements of Statistical Learning Notes</h1>
  
  <p>
    Markdown notes and code snippets for the textbook "Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani and Jeron Friedman
  </p>

</div>

<br />

<!-- Badges -->

## Tools

![Statistics](https://img.shields.io/badge/Statistics-F0A816?style=for-the-badge&logo=data:image/png;base64,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)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

---

<br />

<!-- Table of Contents -->

# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
- [Contact](#handshake-contact)
- [Acknowledgements](#gem-acknowledgements)

<!-- About the Project -->

## :star2: About the Project

Notes taken from studying the textbook. Code snippets for exercises done in Python and notes taken in Markdown

## :handshake: Contact

Author: Martin Ho

Project Link: [https://github.com/minimartzz/ml-design-patterns](https://github.com/minimartzz/ml-design-patterns)

<!-- Acknowledgments -->

## :gem: Acknowledgements
