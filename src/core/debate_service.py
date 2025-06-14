import logging
import re
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.api_client import OpenRouterClient
from src.core.message_formatter import MessageFormatter
from src.core.models import (BetPatternConfig, DebateTopic, DebateTotal,
                         DebateType, DebatorBet, Round, Side)
from src.core.utils import make_rounds


class DebateService:
    def __init__(
        self,
        api_client: OpenRouterClient,
        message_formatter: MessageFormatter,
        bet_pattern_config: BetPatternConfig,
        logger: logging.Logger
    ):
        self.api_client = api_client
        self.message_formatter = message_formatter
        self.bet_pattern_config = bet_pattern_config
        self.logger = logger

    def extract_bet_amount(self, speech_text: str, model: str, round: Round) -> int:
        """
        Extract bet amount from speech text using regex.
        If no valid bet is found, prompts the user for manual input.

        Args:
            speech_text: The model's response text
            model: The model name (for reporting)
            round: The current debate round

        Returns:
            int: The extracted bet amount (0-100)
        """
        bet_pattern = rf"<{self.bet_pattern_config.bet_amount_xml_tag}>(\d+)</{self.bet_pattern_config.bet_amount_xml_tag}>"
        bet_pattern_alternative = rf"<{self.bet_pattern_config.bet_amount_xml_tag.upper()}>(\d+)</{self.bet_pattern_config.bet_amount_xml_tag.upper()}>"
        match = re.search(bet_pattern, speech_text) or re.search(bet_pattern_alternative, speech_text)

        if match:
            bet_amount = int(match.group(1))
            # Ensure bet is within valid range
            return max(0, min(bet_amount, 100))

        # If no valid bet found, ask for manual input
        self.logger.warning(
            f"Could not extract bet from {model} for {round.side.value} {round.speech_type.value}"
        )
        self.logger.info(f"Speech content: {speech_text})")

        while True:
            try:
                user_input = input(
                    f"\nEnter bet amount (0-100) for {model} {round.side.value} {round.speech_type.value}: "
                )
                bet_amount = int(user_input.strip())
                if 0 <= bet_amount <= 100:
                    return bet_amount
                self.logger.error("Bet must be between 0 and 100")
            except ValueError:
                self.logger.error("Please enter a valid number")

    def extract_bet_logic(self, speech_text: str, model: str, round: Round) -> str:
        """
        Extract bet logic from speech text using regex.
        If no valid bet logic is found, prompts the user for manual input.
        Args:
            speech_text: The model's response text
            model: The model name (for reporting)
            round: The current debate round
        Returns:
            str: The extracted bet logic reasoning
        """
        bet_logic_pattern = f"<{self.bet_pattern_config.bet_logic_private_xml_tag}>(.*?)</{self.bet_pattern_config.bet_logic_private_xml_tag}>"
        alternative_bet_logic_pattern = f"<{self.bet_pattern_config.bet_logic_private_xml_tag.upper()}>(.*?)</{self.bet_pattern_config.bet_logic_private_xml_tag.upper()}>"
        match = re.search(bet_logic_pattern, speech_text, re.DOTALL) or re.search(alternative_bet_logic_pattern, speech_text, re.DOTALL)
        if match:
            bet_logic_private = match.group(1).strip()
            return bet_logic_private
        self.logger.warning(
            f"Could not extract bet logic from {model} for {round.side.value} {round.speech_type.value}"
        )
        self.logger.info(f"Speech content: {speech_text}")
        user_input = input(
            f"\nEnter bet logic for {model} {round.side.value} {round.speech_type.value}: "
        )
        return user_input.strip()

    def run_debate(
        self,
        proposition_model: str,
        opposition_model: str,
        motion: DebateTopic,
        path_to_store: Path,
        debate_type: DebateType = DebateType.BASELINE,
    ) -> DebateTotal:
        """
        Run a complete debate including all speeches and judgments

        Args:
            proposition_model: Model for proposition side
            opposition_model: Model for opposition side
            motion: The debate topic
            path_to_store: Path to save debate JSON
            debate_type: Type of debate (BASELINE, PRIVATE_BET, PUBLIC_BET)

        Returns:
            DebateTotal: The completed debate
        """
        self.logger.debug("running debate!")
        self.logger.info(f"Starting debate on motion: {motion.topic_description}")
        self.logger.info(f"Debate type: {debate_type.value}")

        # Initialize debate
        debate = DebateTotal(
            motion=motion,
            debate_type=debate_type,
            proposition_model=proposition_model,
            opposition_model=opposition_model,
            prompts=self.message_formatter.prompts,
            path_to_store=path_to_store,
            debator_bets=[] if debate_type != DebateType.BASELINE else None,
        )

        # Run debate rounds
        rounds = make_rounds()
        for round in rounds:
            self.logger.info(f"Executing round: {round.speech_type} for {round.side}")
            self._execute_round(debate, round)
            debate.save_to_json()

        self.logger.info("Debate completed successfully")
        return debate

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=10, max=20),
        before_sleep=lambda retry_state: print(
            f"Attempt {retry_state.attempt_number} failed. Retrying..."
        ),
    )
    def _execute_round(self, debate: DebateTotal, round: Round):
        """
        Execute a single debate round, handling API communication and state updates
        """
        model = (
            debate.proposition_model
            if round.side == Side.PROPOSITION
            else debate.opposition_model
        )

        messages = self.message_formatter.get_chat_messages(debate, round)

        try:
            response = self.api_client.send_request(model, messages)

            # Store the speech

            # Extract and store bet if not a baseline debate
            if debate.debate_type != DebateType.BASELINE:
                bet_amount = self.extract_bet_amount(response.content, model, round)
                bet_logic_private = self.extract_bet_logic(
                    response.content, model, round
                )
                self.logger.info(
                    f"Extracted bet amount {bet_amount} \n with logic {bet_logic_private}"
                )
                new_bet = DebatorBet(
                    side=round.side,
                    speech_type=round.speech_type,
                    amount=bet_amount,
                    thoughts=bet_logic_private,
                )

                if debate.debator_bets is None:
                    debate.debator_bets = []

                debate.debator_bets.append(new_bet)
                self.logger.info(
                    f"Recorded bet: {bet_amount} for {round.side.value} {round.speech_type.value}"
                )
                cleaned_speech = re.sub(
                    rf"<{self.bet_pattern_config.bet_amount_xml_tag}>\d+</{self.bet_pattern_config.bet_amount_xml_tag}>",
                    "",
                    response.content,
                ).strip()
                cleaned_speech = re.sub(
                    f"<{self.bet_pattern_config.bet_logic_private_xml_tag}>.*?</{self.bet_pattern_config.bet_logic_private_xml_tag}>",
                    "",
                    cleaned_speech,
                    flags=re.DOTALL,
                )

            else:
                cleaned_speech = response.content

            if round.side == Side.PROPOSITION:
                debate.proposition_output.speeches[round.speech_type] = cleaned_speech
            else:
                debate.opposition_output.speeches[round.speech_type] = cleaned_speech

            # Track successful token usage
            debate.debator_token_counts.add_successful_call(
                model=model,
                completion_tokens=response.completion_tokens,
                prompt_tokens=response.prompt_tokens,
                total_tokens=response.completion_tokens + response.prompt_tokens,
            )

            self.logger.info(f"Successfully completed {round.speech_type} for {round.side}")

        except Exception as e:
            print(e)
            if isinstance(e, requests.exceptions.RequestException) and hasattr(
                e, "response"
            ):
                usage = e.response.json().get("usage", {}) if e.response else {}
                debate.debator_token_counts.add_failed_call(
                    model=model,
                    completion_tokens=usage.get("completion_tokens", 0),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
            self.logger.error(f"Error in debate round: {str(e)}")
            raise

    def continue_debate(self, debate_path: Path) -> DebateTotal:
        """
        Continues a partially completed debate by completing any missing speeches.

        Args:
            debate_path: Path to the JSON file containing the partial debate

        Returns:
            DebateTotal: The completed debate
        """
        self.logger.info(f"Continuing debate from {debate_path}")

        # Load the existing debate
        debate = DebateTotal.load_from_json(debate_path)

        # Get all rounds that need to be completed
        rounds = make_rounds()
        incomplete_rounds = []

        for round in rounds:
            if round.side == Side.PROPOSITION:
                speech = debate.proposition_output.speeches[round.speech_type]
            else:
                speech = debate.opposition_output.speeches[round.speech_type]

            if speech == -1:  # Speech is missing
                incomplete_rounds.append(round)

        # Complete missing rounds
        for round in incomplete_rounds:
            self.logger.info(f"Completing missing {round.speech_type} for {round.side}")
            self._execute_round(debate, round)
            debate.save_to_json()

        self.logger.info("Debate continuation completed")
        return debate
