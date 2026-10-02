import streamlit as st
import streamlit.components.v1 as components

# Define custom CSS for styling
custom_css = """
    <style>
        .custom-input input {
            color: #ff6347;  /* Change text color */
            font-size: 18px; /* Change font size */
            font-family: 'Script Bold Italic', cursive;
            padding: 4px;   /* Add padding */
            border-radius: 5px; /* Optional: rounded corners */
            border: 2px solid #000000; /* Optional: border color */
            width: 100%;    /* Set width of the input field */
            box-sizing: border-box; /* Ensure padding and border are included in the width */
        }
    </style>
"""

# Define HTML for the custom input fields
html_input = """
    <div class="custom-input">
        <input type="text" id="name_input" placeholder="Enter your name" />
    </div>
    <div class="custom-input">
        <input type="password" id="pass_input" placeholder="Enter your Password" />
    </div>
    <div class="custom-input">
        <input type="password" id="con_input" placeholder="Confirm your Password" />
    </div>
    <div class="custom-input">
        <input type="email" id="email_input" placeholder="Enter Your Email" />
    </div>
    <div class="custom-input">
        <input type="tel" id="ph_input" placeholder="Enter Your Phone Number" />
    </div>
    <script>
        function submitForm() {
            const name = document.getElementById('name_input').value;
            const password = document.getElementById('pass_input').value;
            const confirmPassword = document.getElementById('con_input').value;
            const email = document.getElementById('email_input').value;
            const phone = document.getElementById('ph_input').value;

            // Redirect with query parameters
            const query = new URLSearchParams({
                name: name,
                password: password,
                confirmPassword: confirmPassword,
                email: email,
                phone: phone
            }).toString();
            window.location.href = `?${query}`;
        }

        // Add event listeners to inputs
        document.querySelectorAll('.custom-input input').forEach(input => {
            input.addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    event.preventDefault(); // Prevent the default Enter key action (like form submission)
                    submitForm(); // Call the submitForm function
                }
            });
        });
    </script>
"""

# Render the HTML with custom CSS
components.html(custom_css + html_input, height=500)

# Extract query parameters
params = st.experimental_get_query_params()

# Retrieve form data from query parameters
if params:
    form_data = {
        "name": params.get("name", [""])[0],
        "password": params.get("password", [""])[0],
        "confirmPassword": params.get("confirmPassword", [""])[0],
        "email": params.get("email", [""])[0],
        "phone": params.get("phone", [""])[0]
    }
    st.write("Form Data Submitted:")
    st.write(form_data)
