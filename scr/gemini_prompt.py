from dotenv import load_dotenv
from google import genai
import os
from google.genai import errors

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


def gemini_output(metrics):
    client = genai.Client(api_key=api_key)

    strategy_block = ""

    for strategy, vals in metrics.items():
        strategy_block += f"""
                    {strategy}
                    Annual Return: {vals['Annual Return']:.4f}
                    Volatility: {vals['Volatility']:.4f}
                    Sharpe Ratio: {vals['Sharpe']:.4f}
                    Max Drawdown: {vals['Max Drawdown']:.4f}
                    """

    prompt = f"""
                You are a quantitative portfolio analyst writing a research note.
                The following results are from an OUT-OF-SAMPLE backtest comparing several portfolio strategies.

                Strategy Metrics
                ----------------
                {strategy_block}

                Tasks:
                1. Identify which strategy performed best overall.
                2. Compare strategies in terms of risk-adjusted performance.
                3. Discuss the return vs volatility tradeoff.
                4. Compare drawdown risk.
                5. Explain what the results suggest about the robustness of the strategies.

                Write the analysis as a structured research report.

                Sections:

                ### Strategy Comparison Report

                #### Executive Summary
                Summarize which strategy performed best overall.

                #### Risk-Adjusted Performance
                Compare Sharpe ratios across strategies.

                #### Return vs Risk Tradeoff
                Explain which strategies provide higher returns versus lower volatility.

                #### Drawdown Analysis
                Compare downside risk across strategies.

                #### Recommendation
                Explain which strategy investors might prefer and under what conditions.

                Important:
                Reference the numerical values when making comparisons.
                Use a professional quantitative finance tone.
                Formatting rules:
                - Write in a professional quantitative research tone.
                - Use clear section headings.
                - Keep explanations concise but insightful.
                - Avoid generic AI phrases.
                - Make the analysis sound like a portfolio manager or quant researcher wrote it.
            """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt)
        return response.text

    except errors.ClientError as e:
        return f"""⚠️ AI analysis currently unavailable.Reason: {e.message if hasattr(e, 'message') else 'API quota or model error.'}
            Please try again later.
            """

    except Exception:
        return "⚠️ AI analysis could not be generated due to an unexpected error.Please try again later."
