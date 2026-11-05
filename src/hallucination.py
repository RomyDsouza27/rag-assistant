
import re


class HallucinationChecker:
    """
    Rule-based support checker.

    Reference counting works for numbered reference lists when
    the complete References section is supplied.
    This is not a complete semantic hallucination detector.
    """

    def check(self, answer, context_chunks):

        if not answer or not answer.strip():
            return {
                "supported": False,
                "score": 0.0,
                "warning": "The generated answer is empty.",
                "unsupported_sentences": []
            }

        if not context_chunks:
            return {
                "supported": False,
                "score": 0.0,
                "warning": "No source context was provided.",
                "unsupported_sentences": [answer.strip()]
            }

        context_parts = []

        for chunk in context_chunks:
            if isinstance(chunk, dict):
                context_parts.append(chunk.get("text", ""))
            else:
                context_parts.append(str(chunk))

        context = "\n".join(context_parts)
        context_lower = context.lower()

        # Special check for numerical reference-count claims
        reference_question_context = re.search(
            r"\b(references|bibliography|works cited)\b",
            context_lower
        )

        count_match = re.search(
            r"\b(\d+)\s+(?:references|citations)\b",
            answer.lower()
        )

        if reference_question_context and count_match:

            # Count numbered entries such as [1], 1., or 1)
            numbered_entries = set(
                re.findall(
                    r"(?m)^\s*(?:\[(\d+)\]|(\d+)[.)])\s+",
                    context
                )
            )

            numbers = {
                int(a or b)
                for a, b in numbered_entries
                if a or b
            }

            if numbers:
                claimed_count = int(count_match.group(1))
                actual_count = len(numbers)

                supported = claimed_count == actual_count

                return {
                    "supported": supported,
                    "score": 1.0 if supported else 0.0,
                    "warning": (
                        "The reference count matches the numbered entries."
                        if supported
                        else (
                            f"The answer claims {claimed_count} references, "
                            f"but {actual_count} numbered entries were detected."
                        )
                    ),
                    "unsupported_sentences": (
                        [] if supported else [answer.strip()]
                    ),
                    "supported_sentences": (
                        [answer.strip()] if supported else []
                    )
                }

        # Original lexical-overlap baseline
        context_normalized = re.sub(
            r"\s+", " ", context_lower
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer.strip()
        )

        supported_sentences = []
        unsupported_sentences = []

        stop_words = {
            "this", "that", "with", "from", "have",
            "been", "were", "they", "their", "there",
            "which", "using", "into", "also", "than",
            "then", "them", "will", "would", "could",
            "should", "about", "because", "system"
        }

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            words = re.findall(
                r"\b[a-zA-Z]{4,}\b",
                sentence.lower()
            )

            keywords = [
                word for word in words
                if word not in stop_words
            ]

            if not keywords:
                supported_sentences.append(sentence)
                continue

            unique_keywords = set(keywords)

            matched = sum(
                1 for word in unique_keywords
                if word in context_normalized
            )

            overlap = matched / len(unique_keywords)

            if overlap >= 0.5:
                supported_sentences.append(sentence)
            else:
                unsupported_sentences.append(sentence)

        total = (
            len(supported_sentences)
            + len(unsupported_sentences)
        )

        score = (
            len(supported_sentences) / total
            if total else 0.0
        )

        return {
            "supported": len(unsupported_sentences) == 0,
            "score": round(score, 2),
            "warning": (
                "Some claims may be unsupported by the context."
                if unsupported_sentences
                else "No unsupported sentences detected by this baseline."
            ),
            "unsupported_sentences": unsupported_sentences,
            "supported_sentences": supported_sentences
        }
