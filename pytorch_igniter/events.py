from ignite.engine.events import Events, CallableEventWithFilter
import re
def event_argument(event):
    if not event:
        return None
    elif isinstance(event, (Events, CallableEventWithFilter)):
        return event
    elif isinstance(event, str):
        m = re.match("^([A-Z_]+)(\\(.*\\))?$", event)
        if not m:
            raise ValueError("Invalid event format: {}. Should be something like \"ITERATION_COMPLETED(every=100)\".".format(event))
        evt = m.group(1)
        if not hasattr(Events, evt):
            raise ValueError("Invalid event name: {}".format(evt))
        evt = getattr(Events, evt)
        filt = m.group(2)
        if not filt:
            return evt
        else:
            m = re.match("\\((.*)=(.*)\\)", filt)
            k = m.group(1)
            v = int(m.group(2))
            kwargs = {k:v}
            return evt(**kwargs)
    else:
        raise ValueError("Event should be string or ignite.engine.events.Event but is {}".format(type(event)))

if __name__=='__main__':
    print(event_argument(None))
    print(event_argument(""))
    print(event_argument(Events.ITERATION_COMPLETED))
    print(event_argument(Events.ITERATION_COMPLETED(every=10)))
    print(event_argument("ITERATION_COMPLETED"))
    print(event_argument("ITERATION_COMPLETED(every=10)"))