def generate_caption_from_grade(grade):
    captions = {
        0: "No signs of osteoarthritis. Normal joint structure and spacing.",
        1: "Doubtful joint space narrowing with possible osteophyte lipping.",
        2: "Definite presence of osteophytes and possible joint space narrowing.",
        3: "Multiple osteophytes and definite joint space narrowing with mild sclerosis.",
        4: "Large osteophytes with severe joint space narrowing and significant sclerosis."
    }
    return captions.get(grade, "Unknown grade")