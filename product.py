class Product:
    def __init__(
        self,
        id: int,
        name: str,
        desc: str,
        price: float,
        cost: float,
        simulation_id: int,
    ) -> None:
        self.id = id
        self.name = name
        self.desc = desc
        self.price = price
        self.cost = cost
        self.simulation_id = simulation_id

    # convert product to prompt format (I define eh)
    def to_prompt_str(self):
        # price is in RM because I am from Malaysia, no plans on foreign currency for now
        return f"(product_id:{self.id},name:{self.name},description:{self.desc},price:RM{self.price})"
