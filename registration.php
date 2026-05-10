<?php
require_once('db.php');

if (isset($_COOKIE['User'])) {
    header('Location: /profile.php');
    exit();
}

$error = '';
$login = '';
$email = '';

if (isset($_POST['submit'])) {
    $link = mysqli_connect($servername, $username, $password, $dbName);

    if (!$link) {
        die('Ошибка подключения: ' . mysqli_connect_error());
    }

    mysqli_set_charset($link, 'utf8mb4');

    $login = trim($_POST['login'] ?? '');
    $email = trim($_POST['email'] ?? '');
    $pass = trim($_POST['password'] ?? '');

    if (!$login || !$email || !$pass) {
       die('input all parameters' ); 
    } 
        $loginSql = mysqli_real_escape_string($link, $login);
        $emailSql = mysqli_real_escape_string($link, $email);
        $passSql = mysqli_real_escape_string($link, $pass);

        $sql = "INSERT INTO users (username, email, password) VALUES ('$loginSql', '$emailSql', '$passSql')";

        if (!mysqli_query($link, $sql)) {
            $error = 'Не удалось добавить пользователя';
        } else {
            mysqli_close($link);
            header('Location: /login.php');
            exit();
        }
    

    mysqli_close($link);
}
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Избосарова Ж.О. - регистрация</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>

<body class="auth-page">

    <div class="pony-card">

        <img src="Gemini_Generated_Image_2gm88z2gm88z2gm8.png" class="pony-avatar" alt="Аватар">

        <h2>Создать аккаунт 🎀</h2>

        <p class="text-muted mb-4">
            Присоединяйся к уютному сообществу
        </p>

        <?php if ($error !== ''): ?>
            <div class="alert alert-danger py-2" role="alert">
                <?php echo htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?>
            </div>
        <?php endif; ?>

        <form action="/registration.php" method="POST" class="d-flex flex-column gap-3">

            <input
                type="text"
                name="login"
                class="form-control"
                placeholder="Логин"
                value="<?php echo htmlspecialchars($login, ENT_QUOTES, 'UTF-8'); ?>"
                required
            >

            <input
                type="email"
                name="email"
                class="form-control"
                placeholder="Email"
                value="<?php echo htmlspecialchars($email, ENT_QUOTES, 'UTF-8'); ?>"
                required
            >

            <input
                type="password"
                name="password"
                class="form-control"
                placeholder="Пароль"
                required
            >

            <button type="submit" name="submit" class="btn btn-pink w-100 mt-2">
                Создать аккаунт ✨
            </button>

        </form>

        <p class="mt-3">
            Already have an account?
            <a href="/login.php" class="magic-link">Login</a>
        </p>

    </div>

</body>
</html>
