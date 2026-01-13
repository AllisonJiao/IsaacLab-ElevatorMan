from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
from typing import Optional, Set, List, Dict, Any

@dataclass
class ElevatorLogEvent:
    tick: int
    event: str
    details: Dict[str, Any]


class ElevatorLogger:
    def __init__(self, print_realtime: bool = True):
        self.events: List[ElevatorLogEvent] = []
        self.print_realtime = print_realtime

    def log(self, tick: int, event: str, **details):
        event_obj = ElevatorLogEvent(
            tick=tick,
            event=event,
            details=details
        )
        self.events.append(event_obj)
        
        # Print to terminal in real-time if enabled
        if self.print_realtime:
            print(f"[t={event_obj.tick:04d}] {event_obj.event}: {event_obj.details}")

    def dump(self):
        """Print all stored events (useful if print_realtime was False)"""
        for e in self.events:
            print(f"[t={e.tick:04d}] {e.event}: {e.details}")


# --------- Enums (matches your notes) ---------

class Direction(Enum):
    UP = auto()
    DOWN = auto()


class DoorStatus(Enum):
    OPENED = auto()
    OPENING = auto()
    CLOSING = auto()
    CLOSED = auto()


# --------- IsaacSim interface (stub) ---------
# Replace these with your actual Isaac Lab / Isaac Sim calls.

class IsaacSimElevatorAPI:
    def go_to_next_floor(self, floor: int) -> None:
        print(f"[IsaacSim] go_to_next_floor({floor})")

    def stop_elevator(self) -> None:
        print("[IsaacSim] stop_elevator()")

    def open_elevator_door(self) -> None:
        print("[IsaacSim] open_elevator_door()")

    def close_elevator_door(self) -> None:
        print("[IsaacSim] close_elevator_door()")


# --------- Inputs each tick ---------

@dataclass
class ElevatorInputs:
    # These are events per tick (set True only on the tick they happen)
    floor_button_pressed: bool = False
    floor_button_floor_num: Optional[int] = None

    open_button_pressed: bool = False
    close_button_pressed: bool = False

    # Event: elevator reports it reached a floor (from sim)
    reached_new_floor: bool = False
    new_floor_num: Optional[int] = None


# --------- Controller ---------

@dataclass
class ElevatorController:
    sim: IsaacSimElevatorAPI
    logger: ElevatorLogger

    tick: int = 0   # global sim time (ticks)

    # Global variables from your notes
    current_floor: int = 0
    next_floor: Optional[int] = None
    elevator_direction: Direction = Direction.UP
    door_status: DoorStatus = DoorStatus.CLOSED

    # Priority queues:
    # - min_heap holds floors above current (ascending)
    # - max_heap holds floors below current (as negative -> max behavior)
    min_floor_heap: list[int] = field(default_factory=list)
    max_floor_heap: list[int] = field(default_factory=list)

    # To avoid duplicates (optional but practical)
    pending_above: Set[int] = field(default_factory=set)
    pending_below: Set[int] = field(default_factory=set)

    # Buffers (tunable)
    DOOR_OPEN_BUFFER_TICKS: int = 60    # "wait" ticks door stays opened before closing
    DOOR_CLOSE_BUFFER_TICKS: int = 30   # "wait" ticks door stays closed after closing
    ELE_BUFFER_TICKS: int = 10          # buffer to prevent instant re-press issues while moving

    # Internal counters/state
    door_open_counter: int = 0
    door_close_counter: int = 0
    ele_buffer_counter: int = 0
    moving: bool = False

    def push_floor_request(self, floor_num: int) -> None:
        """Implements: if floor_num > curr -> min pq, elif < curr -> max pq, else ignore."""
        if floor_num == self.current_floor:
            # If at current floor and door is closed/opening, open door
            if self.door_status in (DoorStatus.CLOSED, DoorStatus.CLOSING):
                self.door_status = DoorStatus.OPENING
                self.door_open_counter = 0
                self.sim.open_elevator_door()
            return

        if floor_num > self.current_floor:
            if floor_num not in self.pending_above:
                heapq.heappush(self.min_floor_heap, floor_num)
                self.pending_above.add(floor_num)
        else:
            if floor_num not in self.pending_below:
                heapq.heappush(self.max_floor_heap, -floor_num)
                self.pending_below.add(floor_num)

    def _pop_next_floor_for_direction(self) -> Optional[int]:
        """Implements: next_floor = (dir==UP ? pop(min) : pop(max)) with direction switching."""
        # Try current direction first
        if self.elevator_direction == Direction.UP:
            if self.min_floor_heap:
                f = heapq.heappop(self.min_floor_heap)
                self.pending_above.discard(f)
                return f
            # Switch direction if current direction is empty
            if self.max_floor_heap:
                self.elevator_direction = Direction.DOWN
                # Recursively call to get floor from new direction
                return self._pop_next_floor_for_direction()
            return None

        elif self.elevator_direction == Direction.DOWN:
            if self.max_floor_heap:
                f = -heapq.heappop(self.max_floor_heap)
                self.pending_below.discard(f)
                return f
            # Switch direction if current direction is empty
            if self.min_floor_heap:
                self.elevator_direction = Direction.UP
                # Recursively call to get floor from new direction
                return self._pop_next_floor_for_direction()
            return None

        return None

    def _can_start_moving(self) -> bool:
        return (
            (not self.moving)
            and self.door_status == DoorStatus.CLOSED
            and self.door_close_counter >= self.DOOR_CLOSE_BUFFER_TICKS
            and self.ele_buffer_counter <= 0
        )

    def _start_move_to_next_floor_if_needed(self) -> None:
        if self.next_floor is None:
            self.next_floor = self._pop_next_floor_for_direction()

        if self.next_floor is None:
            return  # idle

        if self._can_start_moving():
            self.moving = True
            self.ele_buffer_counter = self.ELE_BUFFER_TICKS

            self.logger.log(
                self.tick,
                "ELEVATOR_START_MOVE",
                from_floor=self.current_floor,
                to_floor=self.next_floor,
                direction=self.elevator_direction.name
            )

            self.sim.go_to_next_floor(self.next_floor)

    def _handle_open_button(self) -> None:
        self.logger.log(
            self.tick,
            "DOOR_OPEN_START",
            floor=self.current_floor
        )

        # Notes logic (page 2): match door_status
        if self.door_status == DoorStatus.OPENED:
            # "open_counter = 0" -> extend open time
            self.door_open_counter = 0
            return

        if self.door_status == DoorStatus.OPENING:
            # "door opens more quickly" -> we can reduce open_counter
            self.door_open_counter = max(0, self.door_open_counter - 5)
            return

        if self.door_status == DoorStatus.CLOSED:
            # Prevent opening while elevator is moving / in movement buffer
            if self.moving:
                return
            # If we're still inside close buffer, ignore (like your "no effect" note)
            if self.door_close_counter < self.DOOR_CLOSE_BUFFER_TICKS:
                return
            self.door_status = DoorStatus.OPENING
            self.door_open_counter = 0
            self.sim.open_elevator_door()
            return

        if self.door_status == DoorStatus.CLOSING:
            # Interrupt closing -> open again
            if self.moving:
                return
            self.door_status = DoorStatus.OPENING
            self.door_open_counter = 0
            self.sim.open_elevator_door()

    def _handle_close_button(self) -> None:
        if self.door_status in (DoorStatus.OPENED, DoorStatus.OPENING):
            self.door_status = DoorStatus.CLOSING
            self.door_close_counter = 0
            self.sim.close_elevator_door()
            return

        if self.door_status == DoorStatus.CLOSED:
            # close_counter = DOOR_CLOSE_BUFFER (already closed)
            self.door_close_counter = self.DOOR_CLOSE_BUFFER_TICKS
            return

        if self.door_status == DoorStatus.CLOSING:
            # "door closes more quickly" -> speed it up
            self.door_close_counter += 5

    def _advance_door_state_machine(self) -> None:
        """
        Simple door timing model:
        - OPENING: after a few ticks -> OPENED
        - OPENED: wait DOOR_OPEN_BUFFER -> start CLOSING
        - CLOSING: after a few ticks -> CLOSED and start close buffer
        - CLOSED: count up close buffer
        """
        OPENING_TICKS = 15
        CLOSING_TICKS = 15

        if self.door_status == DoorStatus.OPENING:
            self.door_open_counter += 1
            if self.door_open_counter >= OPENING_TICKS:
                self.door_status = DoorStatus.OPENED

                self.logger.log(
                    self.tick,
                    "DOOR_OPENED",
                    floor=self.current_floor
                )

                self.door_open_counter = 0  # now used as "opened hold" counter

        elif self.door_status == DoorStatus.OPENED:
            self.door_open_counter += 1
            if self.door_open_counter >= self.DOOR_OPEN_BUFFER_TICKS:
                self.door_status = DoorStatus.CLOSING

                self.logger.log(
                    self.tick,
                    "DOOR_CLOSE_START",
                    floor=self.current_floor
                )

                self.door_close_counter = 0
                self.sim.close_elevator_door()

        elif self.door_status == DoorStatus.CLOSING:
            self.door_close_counter += 1
            if self.door_close_counter >= CLOSING_TICKS:
                self.door_status = DoorStatus.CLOSED

                self.logger.log(
                    self.tick,
                    "DOOR_CLOSED",
                    floor=self.current_floor
                )

                # Start close buffer from 0 (will be incremented next tick in CLOSED state)

        elif self.door_status == DoorStatus.CLOSED:
            # accumulate close buffer (increments each tick while closed)
            self.door_close_counter += 1

    def _advance_elevator_buffer(self) -> None:
        if self.ele_buffer_counter > 0:
            self.ele_buffer_counter -= 1

    def update(self, inputs: ElevatorInputs) -> None:
        """
        Call once per sim tick.
        This implements the flow across your page 1–3 notes:
        - accept floor requests (two PQs)
        - open/close buttons control door FSM
        - when door is closed and buffers allow, move to next_floor
        - when reach_new_floor == next_floor: stop, open, wait, close, then continue
        """
        self.tick += 1

        # 1) Floor button event -> enqueue request
        if inputs.floor_button_pressed and inputs.floor_button_floor_num is not None:
            self.logger.log(
                self.tick,
                "FLOOR_BUTTON_PRESSED",
                requested_floor=inputs.floor_button_floor_num,
                current_floor=self.current_floor
            )
            self.push_floor_request(inputs.floor_button_floor_num)

        # 2) Door button events
        if inputs.open_button_pressed:
            self._handle_open_button()
        if inputs.close_button_pressed:
            self._handle_close_button()

        # 3) Elevator reached new floor (event from sim)
        if inputs.reached_new_floor and inputs.new_floor_num is not None:
            self.logger.log(
                self.tick,
                "ELEVATOR_REACHED_FLOOR",
                floor=inputs.new_floor_num,
                expected_floor=self.next_floor
            )

            self.current_floor = inputs.new_floor_num

            # Only process if we reached the expected floor (safety check)
            if self.next_floor is not None and self.current_floor == self.next_floor:
                # Page 3: stop -> open -> wait -> close (FSM handles waits)
                self.sim.stop_elevator()
                self.moving = False

                # Start opening sequence right away
                self.door_status = DoorStatus.OPENING
                self.door_open_counter = 0
                self.sim.open_elevator_door()

                # Consume this target
                self.next_floor = None
            elif self.next_floor is not None and self.current_floor != self.next_floor:
                # Unexpected floor reached - log warning but continue
                self.logger.log(
                    self.tick,
                    "ELEVATOR_UNEXPECTED_FLOOR",
                    reached=inputs.new_floor_num,
                    expected=self.next_floor,
                    warning=True
                )

        # 4) Advance timers/state machines
        self._advance_door_state_machine()
        self._advance_elevator_buffer()

        # 5) If idle + door closed, decide next move
        self._start_move_to_next_floor_if_needed()


# --------- Example usage ---------

if __name__ == "__main__":
    sim = IsaacSimElevatorAPI()
    ctrl = ElevatorController(sim=sim, logger=ElevatorLogger(), current_floor=0, door_status=DoorStatus.CLOSED)

    # Example: user presses 5, then 2
    timeline = [
        ElevatorInputs(floor_button_pressed=True, floor_button_floor_num=5),
        ElevatorInputs(floor_button_pressed=True, floor_button_floor_num=2),
    ]

    # Run some ticks
    print("=" * 60)
    print("Starting Elevator Controller Test")
    print("=" * 60)
    for t in range(5000):
        inp = timeline[t] if t < len(timeline) else ElevatorInputs()
        ctrl.update(inp)

        # Fake sim "reached floor" events for demo:
        # (In real IsaacSim you would set reached_new_floor when your elevator reports it.)
        if ctrl.moving and ctrl.next_floor is not None and t in (20, 80):
            ctrl.update(ElevatorInputs(reached_new_floor=True, new_floor_num=ctrl.next_floor))
    
    # Print summary at the end
    print("=" * 60)
    print(f"Test completed. Final state:")
    print(f"  Current floor: {ctrl.current_floor}")
    print(f"  Next floor: {ctrl.next_floor}")
    print(f"  Direction: {ctrl.elevator_direction.name}")
    print(f"  Door status: {ctrl.door_status.name}")
    print(f"  Moving: {ctrl.moving}")
    print(f"  Pending above: {sorted(ctrl.pending_above)}")
    print(f"  Pending below: {sorted(ctrl.pending_below)}")
    print(f"  Total events logged: {len(ctrl.logger.events)}")
    print("=" * 60)
