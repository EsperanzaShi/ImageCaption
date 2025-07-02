import os
import random
import pandas as pd

clinical_explanations = {
    0: "Healthy knee image.",
    1: "Doubtful joint narrowing with possible osteophytic lipping.",
    2: "Definite presence of osteophytes and possible joint space narrowing.",
    3: "Multiple osteophytes, definite joint space narrowing, with mild sclerosis.",
    4: "Large osteophytes, significant joint narrowing, and severe sclerosis."
}

treatments = {
    0: [
        "You're good! But don't skip leg day.",
        "Your knees are pristine. Flex responsibly.",
        "Keep jogging like you’re running from adulthood.",
        "Healthy joints – a rare flex these days.",
        "Your knees are younger than your playlist.",
        "Treat yourself to a dance party – you've earned it.",
        "No drama in your joints, unlike your group chats.",
        "Knees so clean, even your doctor is jealous.",
        "Don’t mess it up trying to do parkour.",
        "Walk proud – your cartilage is still intact."
    ],
    1: [
        "You're on the OA spectrum now – congrats?",
        "Stretch like your mobility depends on it (it does).",
        "A little creaky, but still street legal.",
        "Your knee is whispering, not screaming – yet.",
        "Avoid heavy squats unless you enjoy regret.",
        "You’re not broken, just... seasoned.",
        "Slap a brace on it and call it a day.",
        "Time to start googling ‘anti-inflammatory smoothies’.",
        "Yoga might help, or at least look cool.",
        "Catch it early – or wait for the fun to begin."
    ],
    2: [
        "Congratulations! You now officially creak when you walk.",
        "You’re halfway to becoming a weather forecaster (via joint pain).",
        "Ibuprofen is your new best friend.",
        "Maybe time to skip the dance battle.",
        "Your knees would like to file a formal complaint.",
        "Look into yoga. Or bubble wrap.",
        "Your joints are like vintage jeans – worn but iconic.",
        "This is not a drill. Unless it's low impact.",
        "Try cycling, but only if you enjoy mild suffering.",
        "Say goodbye to your jump shot dreams."
    ],
    3: [
        "You’ve entered ‘Oh no’ territory.",
        "Your knee’s idea of cardio is standing up.",
        "Try walking… carefully… and with snacks.",
        "Your cartilage wants to resign.",
        "Maybe time to consider a bionic upgrade?",
        "You'll love physical therapy... said no one ever.",
        "You’re now legally allowed to complain about stairs.",
        "Ice packs and sarcasm – your new tools.",
        "Your knee is writing a breakup letter.",
        "You’ve unlocked the ‘grumpy joint’ badge."
    ],
    4: [
        "It's bone-on-bone drama now.",
        "Call your orthopedic surgeon – it’s time.",
        "Your knee just rage-quit cartilage.",
        "Pain is now your part-time job.",
        "Apply for a knee replacement and a vacation.",
        "Bracing for impact – literally.",
        "Your stairs have become a boss battle.",
        "Rely on sarcasm and walking sticks.",
        "Surgery might be cheaper than buying more ice packs.",
        "Time to name your knee – it’s basically sentient now."
    ]
}

output_rows = []

for grade in range(5):
    folder = f"XplainKnee/.knee_data/train/{grade}"
    if not os.path.isdir(folder):
        continue
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        filepath = os.path.join(folder, filename)
        side = "left" if filename.endswith("L.png") or filename.endswith("L.jpg") else "right"
        explanation = clinical_explanations[grade]
        treatment = random.choice(treatments[grade])
        output_rows.append({
            "filepath": filepath,
            "grade": grade,
            "side": side,
            "explanation": explanation,
            "treatment": treatment,
            "split": "test"
        })

df = pd.DataFrame(output_rows)
df.to_csv("XplainKnee/knee_metadata_test.csv", index=False)
print("Saved metadata to XplainKnee/knee_metadata_test.csv")