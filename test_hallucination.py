
from src.hallucination import HallucinationChecker

checker = HallucinationChecker()

context = [
    {
        "text": (
            "This project aims to develop a robust system "
            "for automated language detection using Natural "
            "Language Processing techniques."
        )
    }
]

answer = (
    "The project aims to develop an automated language "
    "detection system using NLP. "
    "The system achieved 99.9% accuracy across 50 languages."
)

result = checker.check(answer, context)

print("\nHallucination Check Results")
print("---------------------------")
print("Supported:", result["supported"])
print("Score:", result["score"])
print("Warning:", result["warning"])
print("Potentially unsupported claims:")

for sentence in result["unsupported_sentences"]:
    print("-", sentence)
