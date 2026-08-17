"""
VisualMind Kafka search-event producer.

"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Producer
from dotenv import load_dotenv


# Load Kafka configuration from the project's .env file.
load_dotenv()


# Kafka broker address for applications running directly on the host.
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

# Kafka topic consumed later by NexusFlow / analytics.
KAFKA_SEARCH_TOPIC = os.getenv(
    "KAFKA_SEARCH_TOPIC",
    "visualmind.search.events",
)


class KafkaSearchProducer:
    """
    Kafka producer responsible for publishing VisualMind search events.
    """

    def __init__(self) -> None:
        """
        Initialize the Kafka producer.
        """

        # Create the Kafka producer once and reuse it for all search events.
        self.producer = Producer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "client.id": "visualmind-api",
            }
        )

    @staticmethod
    def _delivery_report(
        err: Any,
        message: Any,
    ) -> None:
        """
        Callback executed after Kafka accepts or rejects a message.
        """

        if err is not None:
            # Log delivery failures without crashing the search request.
            print(
                "Kafka delivery failed: "
                f"{err}"
            )
            return

        print(
            "Kafka event delivered: "
            f"topic={message.topic()} "
            f"partition={message.partition()} "
            f"offset={message.offset()}"
        )

    def publish_search_event(
        self,
        *,
        query_type: str,
        query_text: str | None,
        result_count: int,
        top_result_id: str | None,
        response_time_ms: float,
    ) -> None:
        """
        Publish a search event to the VisualMind Kafka topic.
        """

        # Build the event using UTC so all downstream analytics use
        # one consistent timezone.
        event = {
            "query_type": query_type,
            "query_text": query_text,
            "result_count": result_count,
            "top_result_id": top_result_id,
            "response_time_ms": round(
                response_time_ms,
                3,
            ),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        # Serialize the event as UTF-8 JSON for Kafka.
        payload = json.dumps(
            event,
            ensure_ascii=False,
        ).encode("utf-8")

        try:
            # Queue the event for asynchronous delivery.
            self.producer.produce(
                topic=KAFKA_SEARCH_TOPIC,
                value=payload,
                callback=self._delivery_report,
            )

            # Serve queued Kafka callbacks without blocking the API
            # waiting for the broker to acknowledge the message.
            self.producer.poll(0)

        except Exception as exc:
            # Kafka analytics must never make a successful product search
            # fail, so producer errors are logged and ignored.
            print(
                "Kafka search event could not be published: "
                f"{exc}"
            )

    def flush(self) -> None:
        """
        Flush any queued Kafka messages.
        """

        self.producer.flush()


# Create one producer instance for the FastAPI process.
kafka_producer = KafkaSearchProducer()