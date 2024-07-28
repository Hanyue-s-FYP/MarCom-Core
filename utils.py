from langchain_core.exceptions import OutputParserException
from typing import Callable

# in case response with valid json, but don't have the field i want
class InvalidJsonException(Exception):
    pass

class FailedAdditionalChekException(Exception):
    pass

# expects chains ending with json parser, invokes the chain until returned response is json and has the expected fields
def get_chain_response_json(chain: any, invoker: dict[str, str], expected_fields: list[str], additional_check: Callable[[dict[str, str]], bool] = None):
    while True:
        try:
            res = chain.invoke(invoker)
            if res is None:
                raise InvalidJsonException
            for k in expected_fields:
                if k not in res:
                    raise InvalidJsonException
            if additional_check is not None:
                if not additional_check(res):
                    raise FailedAdditionalChekException
            return res
        except OutputParserException:
            print("Respond is not in expected format, retrying")
        except InvalidJsonException:
            print("Respond does not have field wanted, retrying", res)
        except FailedAdditionalChekException:
            print("Respond failed additional check, retrying")