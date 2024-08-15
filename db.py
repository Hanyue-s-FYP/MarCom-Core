# responsible to handle all memory and simulation related stuff that requires persistence across restarts
from datetime import datetime
import os
from peewee import *

db = SqliteDatabase(
    os.getenv("DB_FILE") if os.getenv("DB_FILE") is not None else "marcom_simcore.db"
)  # default to marcom_simcore.db

# stores simulation related agent info specific to the simulation (that web server does not store) like rewritten description
class AgentInfo(Model):
    # peewee will automatically handle an auto increment ID field (for uniqueness, should have done with composite keys with agent_id and sim_id but peewee don't have good support on composite key with foreign relationships)
    # these ids are obtained from the web server backend, storing a copy to ensure integrity
    agent_id = IntegerField()
    sim_id = IntegerField()
    rewritten_desc = TextField()
    rewritten_desc_third_person = TextField()

    class Meta:
        database = db

# stores agent memory for that specific agent in that specific simulation
class AgentMemory(Model):
    agent = ForeignKeyField(AgentInfo, backref="memory")
    content = TextField() # storing in plain text for now for simplicity (means when simulating have to make responses into texts and store'em in)
    time_created = DateTimeField(default=datetime.now) # better than just storing a counter and incrementing them to preserve order
    class Meta:
        database = db

# store a copy of the events of the simulation here (also will be forwarded back to web server)
class SimulationEvent(Model):
    agent = ForeignKeyField(AgentInfo, backref="events", null=True) # agents only exist if event type is of ACTION event (eg., BUY/SKIP/MESSAGE, ACTION_RESP)
    sim_id = IntegerField()
    type = TextField() # (BUY/SKIP/MESSAGE): agent takes action, SIMULATION: high level simulation related events, like initializing agent, ACTION_RESP: response to BUY actions of an agent
    content = TextField() # additional information about the event (where the actual message resides) for BUY format is PRODUCT_ID:REASON, for MESSAGE format is AGENT_ID:CONTENT
    cycle = IntegerField() # which cycle does this happen, if is initialisation, then is 0
    time_created = DateTimeField(default=datetime.now) # better than just storing a counter and incrementing them to preserve order
    class Meta:
        database = db
