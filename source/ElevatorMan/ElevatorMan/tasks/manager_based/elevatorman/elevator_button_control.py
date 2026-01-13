from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
from typing import Optional, Set, List, Dict, Any
import threading

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
    # ANSI color codes for terminal output
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    def go_to_next_floor(self, floor: int) -> None:
        print(f"{self.GREEN}[IsaacSim] go_to_next_floor({floor}){self.RESET}")

    def stop_elevator(self) -> None:
        print(f"{self.GREEN}[IsaacSim] stop_elevator(){self.RESET}")

    def open_elevator_door(self) -> None:
        print(f"{self.GREEN}[IsaacSim] open_elevator_door(){self.RESET}")

    def close_elevator_door(self) -> None:
        print(f"{self.GREEN}[IsaacSim] close_elevator_door(){self.RESET}")


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

    # Global variables
    current_floor: int = 1
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

    # Macros
    DOOR_OPEN_TICKS: int = 60    # "wait" ticks door stays opened before closing
    DOOR_CLOSE_TICKS: int = 30   # "wait" ticks door stays closed after closing
    ELE_TICKS: int = 10          # buffer to prevent instant re-press issues while moving
    
    # Movement parameters
    TICKS_PER_FLOOR: int = 100          # Number of ticks to move one floor

    # Internal counters/state
    door_open_counter: int = 0
    door_close_counter: int = 0
    ele_counter: int = 0
    moving: bool = False
    movement_start_tick: Optional[int] = None  # Track when movement started for buffer calculation
    
    # Thread safety for asynchronous button presses
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push_floor_request(self, floor_num: int, thread_safe: bool = False) -> None:
        """
        Implements: if floor_num > curr -> min pq, elif < curr -> max pq, else ignore.
        If within ELE_BUFFER and adjacent floor in current direction, stop immediately.
        
        Args:
            floor_num: The floor number to request
            thread_safe: If True, uses a lock for thread-safe access (for async button presses)
        """
        if thread_safe:
            with self._lock:
                self._push_floor_request_impl(floor_num)
        else:
            self._push_floor_request_impl(floor_num)
    
    def _push_floor_request_impl(self, floor_num: int) -> None:
        """Internal implementation of push_floor_request (without lock)."""
        if floor_num == self.current_floor:
            # Door will only open when:
            # (1) door is CLOSING, or
            # (2) door is CLOSED and elevator is stopped at this floor (within CLOSE_BUFFER)
            if self.door_status == DoorStatus.CLOSING:
                # Door is closing - open it
                self.door_status = DoorStatus.OPENING
                self.door_open_counter = 0
                self.sim.open_elevator_door()
            elif self.door_status == DoorStatus.CLOSED and not self.moving:
                # Door is closed and elevator is stopped - check if within CLOSE_BUFFER
                if self.door_close_counter <= self.DOOR_CLOSE_TICKS:
                    # Within close buffer - open door
                    self.door_status = DoorStatus.OPENING
                    self.door_open_counter = 0
                    self.sim.open_elevator_door()
            return

        # Check if we can stop immediately (within buffer or floor is between current and next)
        # This allows button presses during movement to change next_floor
        # The buffer applies at each floor the elevator passes, not just at movement start
        if self.moving and self.movement_start_tick is not None and self.next_floor is not None:
            ticks_since_movement_start = self.tick - self.movement_start_tick
            
            # Calculate which floor the elevator is currently at/passing
            # Based on movement progress: each floor takes TICKS_PER_FLOOR ticks
            floors_traveled = ticks_since_movement_start // self.TICKS_PER_FLOOR
            if self.elevator_direction == Direction.UP:
                current_passing_floor = self.current_floor + floors_traveled
            else:  # DOWN
                current_passing_floor = self.current_floor - floors_traveled
            
            # Calculate ticks since passing the current floor (for buffer window)
            ticks_since_floor_pass = ticks_since_movement_start % self.TICKS_PER_FLOOR
            is_within_ticks = ticks_since_floor_pass <= self.ELE_TICKS
            
            # Check if requested floor is between current_floor and next_floor in current direction
            is_between_current_and_next = False
            if self.elevator_direction == Direction.UP:
                # Going UP: floor must be > current_floor and < next_floor
                is_between_current_and_next = (self.current_floor < floor_num < self.next_floor)
            elif self.elevator_direction == Direction.DOWN:
                # Going DOWN: floor must be < current_floor and > next_floor
                is_between_current_and_next = (self.current_floor > floor_num > self.next_floor)
            
            # Check if requested floor is adjacent to the floor currently being passed
            is_adjacent_to_passing_floor = False
            if self.elevator_direction == Direction.UP:
                # Going UP: can stop at floor above the current passing floor
                is_adjacent_to_passing_floor = (floor_num == current_passing_floor + 1)
            elif self.elevator_direction == Direction.DOWN:
                # Going DOWN: can stop at floor below the current passing floor
                is_adjacent_to_passing_floor = (floor_num == current_passing_floor - 1)
            
            # Debug logging for buffer checks
            self.logger.log(
                self.tick,
                "ELE_CHECK",
                requested_floor=floor_num,
                current_floor=self.current_floor,
                current_passing_floor=current_passing_floor,
                next_floor=self.next_floor,
                direction=self.elevator_direction.name,
                ticks_since_start=ticks_since_movement_start,
                ticks_since_floor_pass=ticks_since_floor_pass,
                is_within_ticks=is_within_ticks,
                is_between_current_and_next=is_between_current_and_next,
                is_adjacent_to_passing_floor=is_adjacent_to_passing_floor,
                will_stop=(is_within_ticks and is_adjacent_to_passing_floor)
            )
            
            # If within buffer and adjacent to passing floor, stop immediately
            # OR if floor is between current and next, update next_floor to prioritize it
            if is_within_ticks and is_adjacent_to_passing_floor:
                # Immediate stop at adjacent floor
                should_stop = True
            elif is_between_current_and_next and not floor_num == current_passing_floor:
                # Floor is between current and next - update next_floor to service it first
                # This ensures floors in the queue are serviced in order
                should_stop = True
                self.logger.log(
                    self.tick,
                    "ELEVATOR_UPDATE_NEXT_FLOOR",
                    requested_floor=floor_num,
                    previous_next_floor=self.next_floor,
                    reason="floor_between_current_and_next"
                )
            else:
                should_stop = False
            
            if should_stop:
                self.logger.log(
                    self.tick,
                    "ELEVATOR_IMMEDIATE_STOP",
                    requested_floor=floor_num,
                    current_floor=self.current_floor,
                    current_passing_floor=current_passing_floor,
                    previous_next_floor=self.next_floor,
                    ticks_since_start=ticks_since_movement_start,
                    ticks_since_floor_pass=ticks_since_floor_pass,
                    buffer_ticks=self.ELE_TICKS
                )
                # Update next_floor to stop immediately at this floor
                # If there was a previous next_floor, we need to put it back in the queue
                if self.next_floor is not None and self.next_floor != floor_num:
                    # Put the previous target back in the queue
                    prev_floor = self.next_floor
                    if prev_floor > self.current_floor:
                        if prev_floor not in self.pending_above:
                            heapq.heappush(self.min_floor_heap, prev_floor)
                            self.pending_above.add(prev_floor)
                    else:
                        if prev_floor not in self.pending_below:
                            heapq.heappush(self.max_floor_heap, -prev_floor)
                            self.pending_below.add(prev_floor)
                
                # Set next_floor to stop immediately at this floor
                self.next_floor = floor_num
                # Remove from pending sets (if already queued, normal processing will handle it when we reach the floor)
                self.pending_above.discard(floor_num)
                self.pending_below.discard(floor_num)
                return  # Don't add to queue, we're stopping here immediately

        # Normal queue logic: add to appropriate priority queue
        # Use current_floor for comparison (which gets updated incrementally via ELE_BUFFER)
        if floor_num > self.current_floor:
            if floor_num not in self.pending_above:
                heapq.heappush(self.min_floor_heap, floor_num)
                self.pending_above.add(floor_num)
                self.logger.log(
                    self.tick,
                    "FLOOR_ADDED_TO_QUEUE",
                    floor=floor_num,
                    queue="min_floor_heap",
                    queue_contents=list(self.min_floor_heap),
                    current_floor=self.current_floor
                )
        else:
            if floor_num not in self.pending_below:
                heapq.heappush(self.max_floor_heap, -floor_num)
                self.pending_below.add(floor_num)
                self.logger.log(
                    self.tick,
                    "FLOOR_ADDED_TO_QUEUE",
                    floor=floor_num,
                    queue="max_floor_heap",
                    queue_contents=[-f for f in self.max_floor_heap],
                    current_floor=self.current_floor
                )

    def _pop_next_floor_for_direction(self) -> Optional[int]:
        """Implements: next_floor = (dir==UP ? pop(min) : pop(max)) with direction switching."""
        # Try current direction first
        if self.elevator_direction == Direction.UP:
            if self.min_floor_heap:
                self.logger.log(
                    self.tick,
                    "POPPING_FROM_QUEUE",
                    queue="min_floor_heap",
                    queue_contents_before=list(self.min_floor_heap),
                    current_floor=self.current_floor
                )
                f = heapq.heappop(self.min_floor_heap)
                self.pending_above.discard(f)
                self.logger.log(
                    self.tick,
                    "POPPED_FROM_QUEUE",
                    floor=f,
                    queue_contents_after=list(self.min_floor_heap)
                )
                return f
            # Switch direction if current direction is empty
            if self.max_floor_heap:
                self.elevator_direction = Direction.DOWN
                # Recursively call to get floor from new direction
                return self._pop_next_floor_for_direction()
            return None

        elif self.elevator_direction == Direction.DOWN:
            if self.max_floor_heap:
                self.logger.log(
                    self.tick,
                    "POPPING_FROM_QUEUE",
                    queue="max_floor_heap",
                    queue_contents_before=[-f for f in self.max_floor_heap],
                    current_floor=self.current_floor
                )
                f = -heapq.heappop(self.max_floor_heap)
                self.pending_below.discard(f)
                self.logger.log(
                    self.tick,
                    "POPPED_FROM_QUEUE",
                    floor=f,
                    queue_contents_after=[-f for f in self.max_floor_heap]
                )
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
            and self.door_close_counter >= self.DOOR_CLOSE_TICKS
            and self.ele_counter >= self.ELE_TICKS
        )

    def _start_move_to_next_floor_if_needed(self) -> None:
        # Only select next floor if we can actually start moving
        # This ensures we pick the correct floor from the priority queue
        if not self._can_start_moving():
            return  # Not ready to move, don't select floor yet
        
        # Only pop from queue if we don't have a next_floor set
        if self.next_floor is None:
            self.next_floor = self._pop_next_floor_for_direction()

        if self.next_floor is None:
            return  # idle

        # Start moving
        self.moving = True
        self.movement_start_tick = self.tick  # Track when movement starts for buffer calculation
        self.ele_counter = 0

        self.logger.log(
            self.tick,
            "ELEVATOR_START_MOVE",
            from_floor=self.current_floor,
            to_floor=self.next_floor,
            direction=self.elevator_direction.name
        )

        self.sim.go_to_next_floor(self.next_floor)

    def _handle_open_button(self) -> None:

        # Notes logic (page 2): match door_status
        if self.door_status == DoorStatus.OPENED:
            # "open_counter = 0" -> extend open time
            self.door_open_counter = 0
            return

        if self.door_status == DoorStatus.OPENING:
            # -> do nothing
            return

        if self.door_status == DoorStatus.CLOSED:
            # Prevent opening while elevator is moving / in movement buffer
            if self.moving:
                return
            # If we exceed close buffer, ignore
            if self.door_close_counter >= self.DOOR_CLOSE_TICKS:
                return
            # Open door
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
            return

    def _handle_close_button(self) -> None:
        if self.door_status in (DoorStatus.OPENED, DoorStatus.OPENING):
            self.door_status = DoorStatus.CLOSING
            self.door_close_counter = 0
            self.sim.close_elevator_door()
            return

        if self.door_status == DoorStatus.CLOSED:
            # do nothing
            return

        if self.door_status == DoorStatus.CLOSING:
            # do nothing
            return

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
            if self.door_open_counter >= self.DOOR_OPEN_TICKS:
                self.door_status = DoorStatus.CLOSING
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

    def _advance_elevator_ticks(self) -> None:
        """Advance elevator buffer and update current_floor incrementally as elevator passes floors."""
        if self.ele_counter < self.ELE_TICKS:
            self.ele_counter += 1
        
        # ELE_TICKS logic: update current_floor incrementally as elevator passes floors
        # "within ticks: current_floor, exceed buffer: current_floor++"
        # current_floor should be updated to current_passing_floor each time it passes a floor
        if self.moving and self.movement_start_tick is not None and self.next_floor is not None:
            ticks_since_movement_start = self.tick - self.movement_start_tick
            
            # Calculate which floor the elevator is currently at/passing
            floors_traveled = ticks_since_movement_start // self.TICKS_PER_FLOOR
            if self.elevator_direction == Direction.UP:
                current_passing_floor = self.current_floor + floors_traveled
            else:  # DOWN
                current_passing_floor = self.current_floor - floors_traveled
            
            # Calculate ticks since passing the current passing floor
            ticks_since_floor_pass = ticks_since_movement_start % self.TICKS_PER_FLOOR
            
            # If we've exceeded the buffer window for a floor, update current_floor to current_passing_floor
            # This allows the elevator to "forget" floors it has passed
            if ticks_since_floor_pass > self.ELE_TICKS:
                # Exceeded buffer: current_floor = current_passing_floor
                if self.elevator_direction == Direction.UP:
                    if current_passing_floor > self.current_floor:
                        self.current_floor = current_passing_floor
                        self.logger.log(
                            self.tick,
                            "ELEVATOR_UPDATE",
                            floor=self.current_floor
                        )
                else:  # DOWN
                    if current_passing_floor < self.current_floor:
                        self.current_floor = current_passing_floor
                        self.logger.log(
                            self.tick,
                            "ELEVATOR_UPDATE",
                            floor=self.current_floor
                        )
    
    def handle_button_press_async(self, floor_num: int) -> None:
        """
        Thread-safe method to handle button presses asynchronously.
        This can be called from Isaac Sim sensor callbacks or other threads.
        
        Args:
            floor_num: The floor number that was pressed
        """
        self.push_floor_request(floor_num, thread_safe=True)

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

        # 1) Floor button event -> enqueue request (can be called during movement)
        if inputs.floor_button_pressed and inputs.floor_button_floor_num is not None:
            self.logger.log(
                self.tick,
                "FLOOR_BUTTON_PRESSED",
                requested_floor=inputs.floor_button_floor_num,
                current_floor=self.current_floor,
                moving=self.moving
            )
            # Always allow button presses, even during movement
            # The ELE_BUFFER logic will handle immediate stops if applicable
            self.push_floor_request(inputs.floor_button_floor_num, thread_safe=False)

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
                self.movement_start_tick = None  # Reset movement tracking

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
        self._advance_elevator_ticks()

        # 5) If idle + door closed, decide next move
        self._start_move_to_next_floor_if_needed()


# --------- Example usage ---------

if __name__ == "__main__":
    import time
    import signal
    
    sim = IsaacSimElevatorAPI()
    # Set print_realtime=False to reduce logging - only Isaac Sim prints will show
    ctrl = ElevatorController(sim=sim, logger=ElevatorLogger(print_realtime=True), current_floor=0, door_status=DoorStatus.CLOSED)

    # Test timeline: button presses at specific ticks
    # Format: (tick, ElevatorInputs)
    timeline = {
        1: ElevatorInputs(floor_button_pressed=True, floor_button_floor_num=5),
        150: ElevatorInputs(floor_button_pressed=True, floor_button_floor_num=1),
        31: ElevatorInputs(floor_button_pressed=True, floor_button_floor_num=2),
        550: ElevatorInputs(floor_button_pressed=True, floor_button_floor_num=3),
        # Add more test inputs as needed:
        100: ElevatorInputs(open_button_pressed=True),
        200: ElevatorInputs(close_button_pressed=True),
    }

    # Track movement for simulating floor arrivals
    movement_start_tick = None
    TICKS_PER_FLOOR = ctrl.TICKS_PER_FLOOR  # Use the same value from controller config

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n" + "=" * 60)
        print("Shutting down elevator controller...")
        print("=" * 60)
        print(f"Final state:")
        print(f"  Current floor: {ctrl.current_floor}")
        print(f"  Next floor: {ctrl.next_floor}")
        print(f"  Direction: {ctrl.elevator_direction.name}")
        print(f"  Door status: {ctrl.door_status.name}")
        print(f"  Moving: {ctrl.moving}")
        print(f"  Pending above: {sorted(ctrl.pending_above)}")
        print(f"  Pending below: {sorted(ctrl.pending_below)}")
        print(f"  Total events logged: {len(ctrl.logger.events)}")
        print("=" * 60)
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 60)
    print("Starting Elevator Controller - Continuous Loop")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Continuous loop - receives signals from Isaac Sim
    try:
        while True:
            # Get input for current tick (from timeline or empty)
            inp = timeline.get(ctrl.tick, ElevatorInputs())
            
            # Update controller with inputs
            ctrl.update(inp)

            # Simulate floor arrival: when elevator is moving, check if enough time has passed
            if ctrl.moving and ctrl.next_floor is not None:
                # Track when movement started
                if movement_start_tick is None:
                    movement_start_tick = ctrl.tick
                
                # Calculate floors to travel
                floors_to_travel = abs(ctrl.next_floor - ctrl.current_floor)
                ticks_needed = floors_to_travel * TICKS_PER_FLOOR
                
                # Check if we've reached the destination
                if (ctrl.tick - movement_start_tick) >= ticks_needed:
                    # Simulate reaching the floor
                    ctrl.update(ElevatorInputs(
                        reached_new_floor=True,
                        new_floor_num=ctrl.next_floor
                    ))
                    movement_start_tick = None  # Reset for next movement
            else:
                # Not moving, reset movement tracker
                movement_start_tick = None

            # In real Isaac Sim integration, you would:
            # 1. Get button press events from Isaac Sim sensors (can happen asynchronously)
            #    Example: sensor callback -> ctrl.push_floor_request(floor_num, thread_safe=True)
            # 2. Get floor arrival events from elevator position tracking
            # 3. Replace the simulation logic above with actual Isaac Sim data
            #
            # For async button presses from sensors, use:
            #   ctrl.push_floor_request(floor_num, thread_safe=True)
            # This will safely handle button presses even while the elevator is moving
            
            # Small delay to prevent CPU spinning (remove in real integration)
            time.sleep(0.001)  # 1ms delay - adjust as needed
            
    except KeyboardInterrupt:
        signal_handler(None, None)
