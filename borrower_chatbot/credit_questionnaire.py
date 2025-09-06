import random
import re
from dataclasses import dataclass, field

AUG_1_2025 = (2025, 8, 1)  # anchor date (follow the negative-days convention)

GENDER_OPTIONS = ["Male", "Female", "Prefer not to say"]
FAMILY_STATUS_OPTIONS = [
    "Civil marriage", "Married", "Separated", "Single / not married", "Unknown", "Widow"
]
YES_NO = ["Yes", "No"]
ORG_TYPES = [
    "Advertising","Agriculture","Bank","Business Entity Type 1","Business Entity Type 2","Business Entity Type 3",
    "Cleaning","Construction","Culture","Electricity","Emergency","Government","Hotel","Housing",
    "Industry: type 1","Industry: type 10","Industry: type 11","Industry: type 12","Industry: type 13",
    "Industry: type 2","Industry: type 3","Industry: type 4","Industry: type 5","Industry: type 6",
    "Industry: type 7","Industry: type 8","Industry: type 9","Insurance","Kindergarten","Legal Services",
    "Medicine","Military","Mobile","Other","Police","Postal","Realtor","Religion","Restaurant","School",
    "Security","Security Ministries","Self-employed","Services","Telecom",
    "Trade: type 1","Trade: type 2","Trade: type 3","Trade: type 4","Trade: type 5","Trade: type 6","Trade: type 7",
    "Transport: type 1","Transport: type 2","Transport: type 3","Transport: type 4","University","Unknown"
]
EDU_OPTIONS = [
    "Academic degree","Higher education","Incomplete higher","Lower secondary","Secondary / secondary special"
]
INCOME_TYPE_OPTIONS = [
    "Businessman","Commercial associate","Maternity leave","Pensioner","State servant","Student","Unemployed","Working"
]

def _choose_from_list_by_number(user_text: str, options: list[str]):
    """Only allow choosing by number (1-based)."""
    t = user_text.strip()
    if not t.isdigit():
        return None
    idx = int(t) - 1
    if 0 <= idx < len(options):
        return options[idx]
    return None

def _parse_int_allow_separators(val: str):
    """Parse integer while allowing thousand separators like 1,500,000 or 1 500 000."""
    t = re.sub(r"[,_\s]", "", val.strip())
    if not re.fullmatch(r"-?\d+", t):
        return None
    return int(t)

def _parse_float_allow_separators(val: str):
    """Parse float while allowing thousand separators."""
    t = re.sub(r"[,_\s]", "", val.strip())
    try:
        return float(t)
    except Exception:
        return None

def _format_options(options: list[str]):
    return "\n".join([f"{i+1}. {o}" for i, o in enumerate(options)])

def _gen_sk_id_curr():
    return int("9" + str(random.randint(10000, 99999)))  # 6-digit starting with 9

def _gen_ext_source():
    return round(random.uniform(0.2, 0.8), 3)

def _gen_livingarea_medi():
    return round(random.uniform(0.2, 0.8), 3)

def _gen_def30_cnt_social_circle():
    return random.randint(0, 10)

PROMPT_SUFFIX_PICK = '\n\n(Reply with the *option number*. Type "cancel" to stop.)'
PROMPT_SUFFIX_NUMBER = '\n\n(Enter *a number*. Type "cancel" to stop.)'

@dataclass
class CreditQuestionnaire:
    """
    WhatsApp Q&A state machine.
      - start() -> first prompt
      - handle(text) -> (ok: bool, reply: str)
      - is_complete() -> bool
      - build_features() -> dict for /predict (adds hidden features)
    """
    step: int = 0
    answers: dict = field(default_factory=dict)
    finished: bool = False

    def start(self):
        self.step = 1
        intro = (
            "I'll ask a few quick questions for your credit risk prediction.\n"
            "Please follow these rules:\n"
            "• Choose options by NUMBER only (e.g., 1, 2, or 3)\n"
            "• For amounts or days, you can type 1,500,000 or 1500000\n"
            "• Type *cancel* anytime to stop"
        )
        return intro + "\n\n" + self._prompt_for_current_step()

    def is_complete(self):
        return self.finished

    def _prompt_for_current_step(self):
        prompts = {
            1:  ("CODE_GENDER",
                 "What is your gender?\n" + _format_options(GENDER_OPTIONS) + PROMPT_SUFFIX_PICK),
            2:  ("AGE",
                 "What is your age in years? (Two digits, e.g., 18, 30, 50)" + PROMPT_SUFFIX_NUMBER),
            3:  ("NAME_FAMILY_STATUS",
                 "What is your family status?\n" + _format_options(FAMILY_STATUS_OPTIONS) + PROMPT_SUFFIX_PICK),
            4:  ("FLAG_OWN_CAR",
                 "Do you own a car?\n" + _format_options(YES_NO) + PROMPT_SUFFIX_PICK),
            5:  ("OWN_CAR_AGE",
                 "How old is your car in years? (0–60)" + PROMPT_SUFFIX_NUMBER),
            6:  ("FLAG_DOCUMENT_3",
                 "Do you have a valid national ID?\n" + _format_options(YES_NO) + PROMPT_SUFFIX_PICK),
            7:  ("ORGANIZATION_TYPE",
                 "Which best describes your organization type?\n" + _format_options(ORG_TYPES) + PROMPT_SUFFIX_PICK),
            8:  ("DAYS_EMPLOYED",
                 "How many days have you been employed at your current job? (Count from your start date up to today, e.g., 30, 45, 62, 100, 360)" + PROMPT_SUFFIX_NUMBER),
            9:  ("NAME_EDUCATION_TYPE",
                 "What is your highest education level?\n" + _format_options(EDU_OPTIONS) + PROMPT_SUFFIX_PICK),
            10: ("NAME_INCOME_TYPE",
                 "What is your income type?\n" + _format_options(INCOME_TYPE_OPTIONS) + PROMPT_SUFFIX_PICK),
            11: ("AMT_GOODS_PRICE",
                 "What is the price of the item you plan to purchase with this loan? (e.g., valid formats: 1500 or 1,500)" + PROMPT_SUFFIX_NUMBER),
            12: ("AMT_CREDIT",
                 "What total loan amount are you applying for? (e.g., valid formats: 1500 or 1,500)" + PROMPT_SUFFIX_NUMBER),
            13: ("AMT_ANNUITY",
                 "What is the expected monthly installment? (e.g., valid formats: 1500 or 1,500)" + PROMPT_SUFFIX_NUMBER),
            14: ("AMT_INCOME_TOTAL",
                 "What is your total monthly income? (e.g., valid formats: 1500 or 1,500)" + PROMPT_SUFFIX_NUMBER),
            15: ("DAYS_ID_PUBLISH",
                 "How many days ago was your ID last updated? (e.g., 7, 12, 35, 90, 360)" 
                 + PROMPT_SUFFIX_NUMBER
                 + "\n\n⏳ After you send this answer, please wait about 30–60 seconds "
                 "while we calculate your credit risk and generate your report."
             )
        }
        key, text = prompts[self.step]
        return text

    def handle(self, user_text: str):
        if self.finished:
            return True, "✅ Questionnaire already completed."

        try:
            if self.step == 1:
                sel = _choose_from_list_by_number(user_text, GENDER_OPTIONS)
                if not sel:
                    return False, "Please reply with a number 1–3 for gender:\n" + _format_options(GENDER_OPTIONS) + PROMPT_SUFFIX_PICK
                self.answers["CODE_GENDER"] = {"Male": "M", "Female": "F", "Prefer not to say": "XNA"}[sel]
                self.step += 1

            elif self.step == 2:
                n = _parse_int_allow_separators(user_text)
                if n is None or not (10 <= n <= 99):
                    return False, "Please enter a two-digit age (18–99)." + PROMPT_SUFFIX_NUMBER
                
                if n < 18:
                    return False, '🔞 You must be at least 18 years old to apply for a loan.\n\nPlease enter a two-digit age (18–99) or type "cancel" to stop.'
                
                
                self.answers["DAYS_BIRTH"] = int(round(-n * 365.25))
                self.step += 1

            elif self.step == 3:
                sel = _choose_from_list_by_number(user_text, FAMILY_STATUS_OPTIONS)
                if not sel:
                    return False, "Please reply with a number for family status:\n" + _format_options(FAMILY_STATUS_OPTIONS) + PROMPT_SUFFIX_PICK
                self.answers["NAME_FAMILY_STATUS"] = sel
                self.step += 1

            elif self.step == 4:
                sel = _choose_from_list_by_number(user_text, YES_NO)
                if not sel:
                    return False, "Please reply with 1 (Yes) or 2 (No)." + PROMPT_SUFFIX_PICK
                self.answers["FLAG_OWN_CAR"] = {"Yes": "Y", "No": "N"}[sel]
                if sel == "Yes":
                    self.step += 1
                else:
                    self.answers["OWN_CAR_AGE"] = 0
                    self.step = 6

            elif self.step == 5:
                n = _parse_int_allow_separators(user_text)
                if n is None or not (1 <= n <= 60):
                    return False, "Enter a whole number between 1 and 60." + PROMPT_SUFFIX_NUMBER
                self.answers["OWN_CAR_AGE"] = int(n)
                self.step += 1

            elif self.step == 6:
                sel = _choose_from_list_by_number(user_text, YES_NO)
                if not sel:
                    return False, "Please reply with 1 (Yes) or 2 (No)." + PROMPT_SUFFIX_PICK
                self.answers["FLAG_DOCUMENT_3"] = 1 if sel == "Yes" else 0
                self.step += 1

            elif self.step == 7:
                sel = _choose_from_list_by_number(user_text, ORG_TYPES)
                if not sel:
                    return False, "Please reply with a number for organization type:\n" + _format_options(ORG_TYPES) + PROMPT_SUFFIX_PICK
                self.answers["ORGANIZATION_TYPE"] = sel
                self.step += 1

            elif self.step == 8:
                n = _parse_int_allow_separators(user_text)
                if n is None or n < 0:
                    return False, "Enter a positive number of days." + PROMPT_SUFFIX_NUMBER
                self.answers["DAYS_EMPLOYED"] = -abs(n)
                self.step += 1

            elif self.step == 9:
                sel = _choose_from_list_by_number(user_text, EDU_OPTIONS)
                if not sel:
                    return False, "Please reply with a number for education level:\n" + _format_options(EDU_OPTIONS) + PROMPT_SUFFIX_PICK
                self.answers["NAME_EDUCATION_TYPE"] = sel
                self.step += 1

            elif self.step == 10:
                sel = _choose_from_list_by_number(user_text, INCOME_TYPE_OPTIONS)
                if not sel:
                    return False, "Please reply with a number for income type:\n" + _format_options(INCOME_TYPE_OPTIONS) + PROMPT_SUFFIX_PICK
                self.answers["NAME_INCOME_TYPE"] = sel
                self.step += 1

            elif self.step in (11, 12, 13, 14):
                v = _parse_float_allow_separators(user_text)
                if v is None:
                    return False, "Enter a numeric amount (e.g., 1,500,000)." + PROMPT_SUFFIX_NUMBER
                key_map = {11: "AMT_GOODS_PRICE", 12: "AMT_CREDIT", 13: "AMT_ANNUITY", 14: "AMT_INCOME_TOTAL"}
                self.answers[key_map[self.step]] = float(v)
                self.step += 1

            elif self.step == 15:
                n = _parse_int_allow_separators(user_text)
                if n is None or n < 0:
                    return False, "Enter a positive number of days." + PROMPT_SUFFIX_NUMBER
                self.answers["DAYS_ID_PUBLISH"] = -abs(n)
                self.finished = True

            else:
                self.finished = True

        except Exception as e:
            return False, f"Oops, I couldn't process that: {e}"

        if self.finished:
            return True, ""
        return True, self._prompt_for_current_step()

    def build_features(self):
        """Return dict ready for /predict with hidden/auto-generated fields."""
        if not self.finished:
            raise ValueError("Questionnaire not complete.")

        features = dict(self.answers)

        # Hidden/auto-generated
        features["SK_ID_CURR"] = _gen_sk_id_curr()
        features["EXT_SOURCE_1"] = _gen_ext_source()
        features["EXT_SOURCE_2"] = _gen_ext_source()
        features["EXT_SOURCE_3"] = _gen_ext_source()
        features["LIVINGAREA_MEDI"] = _gen_livingarea_medi()
        features["DEF_30_CNT_SOCIAL_CIRCLE"] = _gen_def30_cnt_social_circle()

        return features
