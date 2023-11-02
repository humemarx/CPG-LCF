from datetime import datetime
from datetime import timedelta
from datetime import timezone


def get_version(version):
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )

    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    if version is None:
        return beijing_now.strftime('%Y_%m_%d_%H_%M')
    else:
        return "{}_{}".format(version, beijing_now.strftime('%Y_%m_%d_%H_%M'))