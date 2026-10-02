SEVERITY_ADVICE = {
    "Normal": (
        "Your results look normal. Keep following the 20-20-20 habit, blink fully "
        "when you scroll, and keep screen humidity in check."
    ),
    "Mild": (
        "Results suggest early dry eye. Artificial tears without preservatives and "
        "more deliberate blinking usually help at this stage."
    ),
    "Moderate": (
        "Results suggest moderate dry eye. Add warm compresses, reduce screen hours, "
        "and book an eye specialist appointment if symptoms persist beyond a week."
    ),
    "Severe": (
        "Results point toward severe dry eye. Please see an ophthalmologist soon. "
        "Prescription drops or anti-inflammatory therapy may be needed."
    ),
}

SYMPTOMS = [
    "symptom", "sign", "how do i know", "how to tell",
    "gritty", "itchy", "scratchy", "burning", "stinging", "sore", "painful",
    "blurred", "blurry", "watery", "tired", "fatigue", "burnout",
    "my eyes hurt", "eyes are red",
]

TRIGGERS = [
    "cause", "why", "reason", "trigger", "risk factor", "prevent", "happen",
    "screen", "computer", "laptop", "mobile", "phone", "monitor", "tv", "reading",
    "driving", "night", "air condition", "ac", "wind", "humid", "low humidity",
    "airport", "smoke", "dust", "pollution", "contact lens", "contacts", "lens",
]

TREATMENTS = [
    "treatment", "treat", "medicine", "medication", "drop", "drops", "artificial tear",
    "therapy", "cure", "remedy", "home remedy", "exercise",
    "diet", "food", "vitamin", "omega", "water", "sleep", "blink",
]

DOCTOR = [
    "doctor", "specialist", "ophthalmologist", "optometrist", "hospital", "clinic",
    "diagnos", "test", "checkup", "check up", "appointment", "see someone",
]

STAGES = ["stage", "mild", "moderate", "severe", "normal", "result", "stage 1", "stage 2"]

GENERAL = [
    "hello", "hi", "hey", "good morning", "good evening", "how are you", "thanks",
    "thank you", "bye", "who are you", "what can you do", "help",
]

DISCLAIMER = (
    "This assistant gives general information only. It is not a medical diagnosis. "
    "For persistent or severe symptoms, consult an ophthalmologist."
)


DISCLAIMER = (
    "This assistant gives general information only. It is not a medical diagnosis. "
    "For persistent or severe symptoms, consult an ophthalmologist."
)


def _contains(text, keywords):
    return any(word in text for word in keywords)


class DryEyeChatbot:
    """Keyword-routed dry eye assistant.

    Answers are static clinical guidance keyed on the user's latest screening stage,
    which the Streamlit layer passes in from session state. It deliberately holds no
    trained estimator: that would mean a pickled model that cannot be unpickled
    across scikit-learn versions, and the stage is already known from the survey.
    """

    def __init__(self, dataset_path="Dataset.xlsx"):
        self.dataset_path = dataset_path
        self.last_result = None

    def set_result(self, stage):
        self.last_result = stage

    def respond(self, message, stage=None):
        text = (message or "").lower().strip()
        if not text:
            return "Please type a question about dry eye disease."

        if stage:
            self.last_result = stage

        if _contains(text, GENERAL):
            return (
                "Hello. I am your dry eye assistant. I can explain symptoms, causes, "
                "prevention, treatments, and help you understand your screening result. "
                f"{DISCLAIMER}"
            )

        if _contains(text, STAGES) and self.last_result:
            stage_name = self.last_result
            advice = SEVERITY_ADVICE.get(
                stage_name, "Keep monitoring your symptoms and stay hydrated."
            )
            return (
                f"Your screening result is **{stage_name}**.\n\n"
                f"{advice}\n\n{STAGE_FACTS.get(stage_name, '')}"
            )

        if _contains(text, STAGES):
            return (
                "I do not have a result for this session yet. Run a screening first "
                "and I can explain what it means."
            )

        if _contains(text, DOCTOR):
            return (
                "You should see an ophthalmologist or optometrist if symptoms persist "
                "more than a week, if your eyes are painful or very red, or if vision "
                "changes. They can measure tear film breakup time, check for corneal "
                "damage, and prescribe the right drops.\n\n"
                "Dry eye can also be a symptom of Sjogren's syndrome or a side effect "
                "of many medications, so a doctor should rule those out."
            )

        if _contains(text, TREATMENTS):
            return (
                "Common treatments for dry eye:\n\n"
                "- Preservative-free artificial tears, used several times a day\n"
                "- Lubricating ointment at night for severe dryness\n"
                "- Warm compresses for 5-10 minutes to loosen oily gland secretions\n"
                "- Omega-3 fatty acids and staying hydrated\n"
                "- Eyedrops such as cyclosporine or lifitegrast for moderate to severe cases\n"
                "- Steroid eye drops for short-term control of inflammation\n\n"
                "Use preservative-free drops if you need them more than four times a day."
            )

        if _contains(text, TRIGGERS):
            return (
                "Common dry eye triggers:\n\n"
                "- Long screen time, which cuts blink rate by roughly half\n"
                "- Air conditioning and heating vents pointing at your face\n"
                "- Wind, dust, smoke, and low-humidity environments\n"
                "- Contact lens wear, which dehydrates the cornea\n"
                "- Dehydration, poor sleep, and low blink completeness\n\n"
                "Fix the biggest one first: screen breaks and airflow direction usually "
                "give the fastest improvement."
            )

        if _contains(text, SYMPTOMS):
            return (
                "Typical dry eye symptoms:\n\n"
                "- Grittiness, itchiness, or a scratchy sensation\n"
                "- Burning or stinging that worsens through the day\n"
                "- Redness, soreness, or a feeling of grit after screen time\n"
                "- Blurry vision that clears when you blink\n"
                "- Watery eyes, which is a reflex response to irritation\n\n"
                "Tell me about your symptoms and I can suggest what to do next."
            )

        return (
            "I can help with dry eye symptoms, causes, prevention, treatments, and "
            "explaining your screening result. Try asking something like "
            "\"what causes dry eye\", \"how do I treat it\", or \"what does my result mean\"."
            f"\n\n{DISCLAIMER}"
        )


STAGE_FACTS = {    "Normal": "OSDI is the Ocular Surface Disease Index, a validated 6-question survey.",
    "Mild": "Early dry eye is often fully reversible with habit changes.",
    "Moderate": "At this stage, adding a doctor-prescribed anti-inflammatory drop helps.",
    "Severe": "Severe cases may need cyclosporine, punctal plugs, or corticosteroid drops.",
}