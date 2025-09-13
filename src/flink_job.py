import collections
import collections.abc

for _abc in ("Mapping", "MutableMapping", "Sequence", "MutableSequence"):
    if not hasattr(collections, _abc):
        setattr(collections, _abc, getattr(collections.abc, _abc))

import json
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types

def main():
    
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    
    kafka_props = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'flink-click-group'
    }
    kafka_consumer = FlinkKafkaConsumer(
        topics='movie-clicks',
        deserialization_schema=SimpleStringSchema(),       
        properties=kafka_props
    )
    kafka_consumer.set_start_from_latest()

    
    stream = env.add_source(kafka_consumer)

    parsed = stream.map(
        lambda x: json.loads(x),
        output_type=Types.MAP(Types.STRING(), Types.STRING())
    )

    
    updated = parsed.map(
        lambda evt: {
            'userId': evt['userId'],
            'liked': [int(evt['movieId'])]
        },
        output_type=Types.MAP(Types.STRING(), Types.PICKLED_BYTE_ARRAY())
    )

    
    updated.print()

    env.execute("Movie Likes State Updater")

if __name__ == '__main__':
    main()