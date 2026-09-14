"""Run inference with a locally trained fallacy classifier."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(sentences, model_path):
    """Predict the fallacy label and confidence for each sentence."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)

    confidences, label_ids = probabilities.max(dim=-1)

    return [
        {
            "text": text,
            "label": model.config.id2label[label_id.item()],
            "confidence": confidence.item(),
        }
        for text, label_id, confidence in zip(sentences, label_ids, confidences)
    ]


if __name__ == "__main__":
    model_path = Path("checkpoints/best_model/8wfncto3")
    sentences = [
        # Expected: ad_populum
        "Everyone supports this policy, so it must be correct.",
        # Expected: ad_hominem
        "You cannot trust her argument because she is lazy.",
        # Expected: false_dilemma
        "Either we ban all cars or climate change will continue forever.",
        # Expected: appeal_to_authority
        "A famous physicist says this diet reverses diabetes, so the clinical trials questioning it must be wrong.",
        # Expected: false_causality
        "Crime fell after the city installed surveillance cameras, which proves the cameras caused the decline.",
        # Expected: cherry_picking
        "The company highlighted three satisfied customers while ignoring a survey in which most users reported problems.",
        # Expected: slippery_slope
        """At the school board meeting, the proposal sounded limited: allow final-year
        students to check their phones during the lunch break. But approving even that
        small exception will make the existing rule impossible to defend. Students will
        soon demand phones between classes, then during lessons, and teachers who object
        will be accused of treating them unfairly. Once phones are common in class,
        nobody will take notes or listen. Grades will lose their meaning, universities
        will stop trusting our diplomas, and employers will avoid hiring graduates from
        this district. Families will move away, funding will disappear, and the schools
        will eventually close. A lunch-break exception therefore puts the future of the
        entire community at risk, so the board must preserve the total ban.""",
        # Expected: appeal_to_authority
        """Investment Committee Memo: The retirement fund is considering a large stake
        in a battery startup whose financial statements and safety tests remain
        incomplete. We should nevertheless approve the investment immediately. A Nobel
        Prize-winning theoretical physicist praised the company during a television
        interview and said its founder understands the future of energy. Someone with
        that level of intelligence and scientific reputation would not support an
        unreliable business. The accountants raising concerns have never won major
        awards, and the engineers requesting more tests are unknown outside their own
        laboratories. Their reservations should not outweigh the judgment of a person
        recognized around the world. Because this celebrated scientist believes in the
        company, the technology must be sound and the investment must be safe.""",
        # Expected: cherry_picking
        """The annual sustainability report declares the efficiency program an
        unquestionable success. It highlights two offices that reduced electricity use
        during unusually mild months, one factory that recycled more packaging, and
        three enthusiastic comments from an employee survey. The report leaves out the
        other eighteen factories, where total energy consumption increased, and does not
        discuss the independent audit that found repeated waste-handling violations. It
        also omits hundreds of survey responses describing broken equipment and missed
        targets. Management argues that these broader figures should not distract from
        the locations that improved. Since the selected offices, factory, and comments
        all show progress, the program is clearly effective across the whole company and
        requires no changes.""",
        # Touché test - Expected: appeal_to_authority
        """Yeah, yeah, yeah, whatever... All your "Joker is only considered a
        masterpiece because audiences are dumbed down by mainstream movies" argument is
        countered easily by one thing - Golden Lion in Venice. These are not the kind of
        people who watch only mainstream movies, and they selected Joker over any other
        movie at the festival. So, your argument is invalid from the start.""",
        # CoCoLoFa test - Expected: slippery_slope
        """It is good to see that an unsolved murder seems to be closer to being solved.
        I am concerned with all the jubilation over someone being charged in this case.
        Everyone seems to be acting as this person is guilty when it is just a charge. It
        is very frightening when we are immediately assumed to be guilty on just an
        accusation. What comes next after this? Will we just put people in jail for life
        upon being accused? After that seems to be accepted, will we just execute them
        before any trial? I understand people wanting justice but we must be careful when
        throwing out systems that protect the innocent. I hope we will all think about
        this and have discussion with others to make sure we don't lead ourselves down
        this dark path.""",
        # CoCoLoFa test - Expected: hasty_generalization
        """Can't say I'm surprised by this, at least when it comes to the nature of the
        Japanese justice system. Anytime I see an article on the subject, it's always
        talking about how harsh and unfair it is. The simple fact I've never seen anything
        good written about it probably suggests it's irredeemable. I hope Japan is able to
        do something about this.""",
        # CoCoLoFa test - Expected: false_dilemma
        """I think if you're running a country like this, you either criminalize
        investigative reporting like the kind done by Kuciak, or you allow it but make no
        guarantees about the personal safety of such journalists. You can either make it
        impossible for them to get themselves killed or you can strongly discourage them
        from engaging in work that might get themselves killed. What other options could
        there be?""",
    ]

    for prediction in predict(sentences, model_path):
        print(f"{prediction['label']} ({prediction['confidence']:.2%})")
        print(f"  {prediction['text']}\n")
