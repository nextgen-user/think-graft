# Answer key: postmortem (planted in make_tasks.py)
Root cause: the 01:50 deploy web v2.3.1 turned on audit.verbose=true (full request bodies); audit.log grew ~0.6 GB/min
and filled /var (61% -> 100% by 02:11). The database lives on /var, so every order write failed with "No space left on
device" from 02:11 to 02:41 (30 minutes): 54 failed orders, checkout healthchecks 500.
Recovery: on-call (paged 02:38) deleted old audit logs at 02:41 (38 GB) - a temporary fix: the flag was still on and
/var filled again (64% -> ~85% by 03:05) until v2.3.2 reverted audit.verbose at 03:05. The incident really ended at 03:05,
and it would have recurred without the revert.
Red herrings: the CDN 502s from edge fra-2 at 02:05-02:07 (static assets, healed by themselves, before the db errors);
the cache warm-up miss warning at 02:03 (expected after a deploy). Neither caused the outage.
Good action items: disk-usage alerting on /var well before 100% (paging took 27 min), separate log and db volumes, log
rotation/size caps for audit logs, review config changes in deploys (a "copy fix" release shipped a risky flag), alert on
order failure rate, runbook: revert the flag rather than delete logs.
